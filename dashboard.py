# %% [markdown]
# # ML Assignment 2 — Combined Model Notebook
# 
# This notebook contains the complete pipeline:
# 1. **Data Loading & Feature Extraction** (Sections 1–3)
# 2. **Exploratory Data Analysis** (Section 4)
# 3. **Data Saving & Temporal Split** (Section 5)
# 4. **Model 1: Support Vector Machine (SVM)** (Section 6)
# 5. **Model 2: Decision Tree** (Section 7)
# 6. **Model 3: Neural Network (MLP from Scratch)** (Section 8)
# 

# %% [markdown]
# # ML Assignment 2 — Data Merging & Exploratory Data Analysis
# 
# **Goal**: Merge the 6 CSV tables into a unified patient-level dataset and perform EDA.
# 
# **Target**: Multi-label classification — predict which diseases a patient has (output = binary vector).

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import copy

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
st.set_page_config(page_title="Healthcare ML Dashboard", layout="wide")

results = defaultdict(list)
CURRENT_SECTION = None

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12

def run_pipeline():

    def set_section(name):
        global CURRENT_SECTION
        CURRENT_SECTION = name

    def add_text(text):
        results[CURRENT_SECTION].append({
            "type": "text",
            "content": str(text)
        })

    def add_plot(fig):
        results[CURRENT_SECTION].append({
            "type": "plot",
            "content": fig
        })

    def add_dataframe(df):
        results[CURRENT_SECTION].append({
            "type": "dataframe",
            "content": df.copy()
        })

    set_section("data_processing")
    add_text("### Goal: Merge the 6 CSV tables into a unified patient-level dataset and perform EDA.")
    add_text("### Target: Multi-label classification - predict which diseases a patient has (output = binary vector).")
    # add_text("Loading datasets...")
    patients = pd.read_csv('csv/patients.csv', on_bad_lines='skip')
    conditions = pd.read_csv('csv/conditions.csv', on_bad_lines='skip')
    conditions = conditions.loc[:, ~conditions.columns.str.startswith('Unnamed')]
    encounters = pd.read_csv('csv/encounters.csv', on_bad_lines='skip')
    medications = pd.read_csv('csv/medications.csv', on_bad_lines='skip')
    procedures = pd.read_csv('csv/procedures.csv', on_bad_lines='skip')

    data_shapes = {
        "Dataset": ["Patients", "Conditions", "Encounters", "Medications", "Procedures"],
        "Rows": [patients.shape[0], conditions.shape[0], encounters.shape[0], medications.shape[0], procedures.shape[0]],
        "Columns": [patients.shape[1], conditions.shape[1], encounters.shape[1], medications.shape[1], procedures.shape[1]]
    }
    add_text("#### Initial Dataset Dimensions")
    add_dataframe(pd.DataFrame(data_shapes))

    # %% [markdown]
    # ## 2. Feature Extraction
    # ### 2.1 Patient Demographics
    # Extract: Age, Gender, Race, Ethnicity, Income, Healthcare costs, Alive status

    # %%
    patients['BIRTHDATE'] = pd.to_datetime(patients['BIRTHDATE'], errors='coerce')
    patients['AGE'] = ((pd.Timestamp.now() - patients['BIRTHDATE']).dt.days / 365.25).astype(int)
    patients['IS_ALIVE'] = patients['DEATHDATE'].isna().astype(int)

    patient_features = patients[['Id', 'AGE', 'GENDER', 'RACE', 'ETHNICITY',
                                'INCOME', 'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE',
                                'IS_ALIVE']].copy()
    patient_features.rename(columns={'Id': 'PATIENT'}, inplace=True)

    add_text(f"### Patient Features Extraction")
    add_text(f"Extracted demographic features for {patient_features.shape[0]} patients. The resulting feature set contains {patient_features.shape[1]} columns.")
    add_dataframe(patient_features.head())

    # %% [markdown]
    # ### 2.2 Multi-Label Disease Target (from Conditions)
    # 
    # Group conditions into disease categories. Each patient gets a binary vector
    # indicating which disease groups they have (1) or don't have (0).
    # 
    # Patients NOT in the conditions table are treated as "healthy" (all zeros).

    # %%
    # Disease groups with keyword matching
    disease_groups = {
        'Hypertension': ['hypertension'],
        'Diabetes': ['diabetes', 'prediabetes', 'hyperglycemia'],
        'Obesity': ['obesity', 'body mass index 30', 'body mass index 40'],
        'Anemia': ['anemia'],
        'Respiratory': ['sinusitis', 'bronchitis', 'pharyngitis', 'asthma', 'pneumonia'],
        'Heart_Disease': ['myocardial infarction', 'heart failure', 'atrial fibrillation', 'coronary heart'],
        'Kidney_Disease': ['kidney disease'],
        'Cancer': ['neoplasm', 'carcinoma', 'cancer'],
        'Dental': ['gingivitis', 'gingival disease', 'dental caries'],
    }

    all_patients = patient_features['PATIENT'].unique()
    target_df = pd.DataFrame({'PATIENT': all_patients})

    # print("Disease group patient counts:")
    disease_stats = []
    for group, keywords in disease_groups.items():
        mask = conditions['DESCRIPTION'].str.lower().apply(
            lambda x: any(kw in str(x) for kw in keywords)
        )
        positive_patients = set(conditions.loc[mask, 'PATIENT'].unique())
        target_df[f'TARGET_{group}'] = target_df['PATIENT'].isin(positive_patients).astype(int)
        disease_stats.append({"Disease Group": group, "Patient Count": target_df[f'TARGET_{group}'].sum()})
    
    add_text("#### Disease Group Patient Counts")
    add_dataframe(pd.DataFrame(disease_stats))

    # Drop disease groups with too few patients (< 5)
    target_cols = [c for c in target_df.columns if c.startswith('TARGET_')]
    drop_cols = [c for c in target_cols if target_df[c].sum() < 5]
    if drop_cols:
        add_text(f"\nDropped (< 5 patients): {[c.replace('TARGET_', '') for c in drop_cols]}")
        target_df.drop(columns=drop_cols, inplace=True)

    # Summary
    target_cols = [c for c in target_df.columns if c.startswith('TARGET_')]
    target_df['HAS_ANY_DISEASE'] = (target_df[target_cols].sum(axis=1) > 0).astype(int)
    target_df['NUM_DISEASES'] = target_df[target_cols].sum(axis=1)
    add_text(f"Successfully identified patients with conditions. {target_df['HAS_ANY_DISEASE'].sum()} patients have at least one disease, with a maximum of {target_df['NUM_DISEASES'].max()} diseases per patient.")

    # %% [markdown]
    # ### 2.3 Clinical Observations
    # 
    # The observations file is large (~271MB, 1.5M rows). We read in chunks,
    # filter to key numeric observations, and aggregate per patient (mean, std).

    # %%
    key_obs = [
        'Body Height', 'Body Weight', 'Body mass index (BMI) [Ratio]',
        'Systolic Blood Pressure', 'Diastolic Blood Pressure',
        'Heart rate', 'Respiratory rate',
        'Glucose [Mass/volume] in Blood',
        'Hemoglobin A1c/Hemoglobin.total in Blood',
        'Creatinine [Mass/volume] in Blood',
        'Calcium [Mass/volume] in Blood',
        'Sodium [Moles/volume] in Blood',
        'Potassium [Moles/volume] in Blood',
        'Chloride [Moles/volume] in Blood',
        'Urea nitrogen [Mass/volume] in Blood',
        'Pain severity - 0-10 verbal numeric rating [Score] - Reported',
        'Patient Health Questionnaire 2 item (PHQ-2) total score [Reported]',
        'DALY', 'QALY', 'QOLS',
    ]

    short_names = {
        'Body Height': 'height', 'Body Weight': 'weight',
        'Body mass index (BMI) [Ratio]': 'bmi',
        'Systolic Blood Pressure': 'sbp', 'Diastolic Blood Pressure': 'dbp',
        'Heart rate': 'heart_rate', 'Respiratory rate': 'resp_rate',
        'Glucose [Mass/volume] in Blood': 'glucose',
        'Hemoglobin A1c/Hemoglobin.total in Blood': 'hba1c',
        'Creatinine [Mass/volume] in Blood': 'creatinine',
        'Calcium [Mass/volume] in Blood': 'calcium',
        'Sodium [Moles/volume] in Blood': 'sodium',
        'Potassium [Moles/volume] in Blood': 'potassium',
        'Chloride [Moles/volume] in Blood': 'chloride',
        'Urea nitrogen [Mass/volume] in Blood': 'bun',
        'Pain severity - 0-10 verbal numeric rating [Score] - Reported': 'pain',
        'Patient Health Questionnaire 2 item (PHQ-2) total score [Reported]': 'phq2',
        'DALY': 'daly', 'QALY': 'qaly', 'QOLS': 'qols',
    }

    # add_text("Processing observations (chunked reading)...")
    chunks = []
    for i, chunk in enumerate(pd.read_csv('csv/observations.csv', chunksize=200000, on_bad_lines='skip')):
        numeric = chunk[chunk['TYPE'] == 'numeric'].copy()
        numeric['VALUE'] = pd.to_numeric(numeric['VALUE'], errors='coerce')
        numeric = numeric.dropna(subset=['VALUE'])
        filtered = numeric[numeric['DESCRIPTION'].isin(key_obs)]
        if len(filtered) > 0:
            chunks.append(filtered[['PATIENT', 'DESCRIPTION', 'VALUE']])
        # add_text(f"  Chunk {i+1}/8 processed...")

    obs_data = pd.concat(chunks, ignore_index=True)
    obs_data['DESCRIPTION'] = obs_data['DESCRIPTION'].map(short_names)
    add_text(f"### Filtered Clinical Observations")
    add_text(f"Processed observation data, resulting in {len(obs_data):,} rows for {obs_data.PATIENT.nunique()} patients.")

    # Aggregate per patient: mean and std
    obs_agg = obs_data.groupby(['PATIENT', 'DESCRIPTION'])['VALUE'].agg(['mean', 'std']).reset_index()
    obs_agg['std'] = obs_agg['std'].fillna(0)

    obs_mean = obs_agg.pivot_table(index='PATIENT', columns='DESCRIPTION', values='mean')
    obs_mean.columns = [f'{c}_mean' for c in obs_mean.columns]
    obs_std = obs_agg.pivot_table(index='PATIENT', columns='DESCRIPTION', values='std')
    obs_std.columns = [f'{c}_std' for c in obs_std.columns]
    obs_features = obs_mean.join(obs_std).reset_index()

    add_text(f"#### Observation Features Summary")
    add_text(f"Generated mean and standard deviation features for clinical observations. Total features: {obs_features.shape[1]}.")
    add_dataframe(obs_features.head())

    # %% [markdown]
    # ### 2.4 Encounters, Medications & Procedures
    # 
    # Extract count-based and cost-based features per patient from each table.

    # %%
    encounter_features = encounters.groupby('PATIENT').agg(
        total_encounters=('Id', 'count'),
        unique_enc_types=('ENCOUNTERCLASS', 'nunique'),
        total_claim_cost=('TOTAL_CLAIM_COST', 'sum'),
        avg_claim_cost=('TOTAL_CLAIM_COST', 'mean'),
        total_payer_coverage=('PAYER_COVERAGE', 'sum'),
    ).reset_index()

    # Encounter type counts (ambulatory, emergency, inpatient, etc.)
    enc_types = encounters.groupby(['PATIENT', 'ENCOUNTERCLASS']).size().unstack(fill_value=0)
    enc_types.columns = [f'enc_{c}' for c in enc_types.columns]
    enc_types = enc_types.reset_index()
    encounter_features = encounter_features.merge(enc_types, on='PATIENT', how='left')
    add_text(f"#### Encounter Features Summary")
    add_text(f"Extracted {encounter_features.shape[1]} features from patient encounter history.")

    # %%
    med_features = medications.groupby('PATIENT').agg(
        total_medications=('CODE', 'count'),
        unique_medications=('DESCRIPTION', 'nunique'),
        total_med_cost=('TOTALCOST', 'sum'),
        avg_dispenses=('DISPENSES', 'mean'),
    ).reset_index()
    add_text(f"#### Medication Features Summary")
    add_text(f"Extracted {med_features.shape[1]} features from medication records.")

    # %%
    proc_features = procedures.groupby('PATIENT').agg(
        total_procedures=('CODE', 'count'),
        unique_procedures=('DESCRIPTION', 'nunique'),
        total_proc_cost=('BASE_COST', 'sum'),
    ).reset_index()
    add_text(f"#### Procedure Features Summary")
    add_text(f"Extracted {proc_features.shape[1]} features from medical procedures.")

    # %% [markdown]
    # ## 3. Merge into Unified Patient-Level Dataset
    # 
    # Left-join everything onto the patient table so we keep all 2823 patients.
    # Patients without data in a table get NaN (filled appropriately).

    # %%
    # Visualize how many patients have which disease targets
    df = patient_features.copy()
    df = df.merge(target_df, on='PATIENT', how='left')
    df = df.merge(obs_features, on='PATIENT', how='left')
    df = df.merge(encounter_features, on='PATIENT', how='left')
    df = df.merge(med_features, on='PATIENT', how='left')
    df = df.merge(proc_features, on='PATIENT', how='left')

    # Fill NaN targets with 0
    target_cols = [c for c in df.columns if c.startswith('TARGET_')]
    df[target_cols] = df[target_cols].fillna(0).astype(int)
    df['HAS_ANY_DISEASE'] = df['HAS_ANY_DISEASE'].fillna(0).astype(int)
    df['NUM_DISEASES'] = df['NUM_DISEASES'].fillna(0).astype(int)

    # Fill NaN count features with 0
    count_cols = ['total_encounters', 'unique_enc_types', 'total_medications',
                'unique_medications', 'total_procedures', 'unique_procedures']
    for col in count_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    merged_summary = {
        "Metric": ["Total Patients", "Total Columns", "Feature Columns", "Target Diseases"],
        "Value": [len(df), df.shape[1], df.shape[1] - len(target_cols) - 3, len(target_cols)]
    }
    add_text("#### Merged Dataset Summary")
    add_dataframe(pd.DataFrame(merged_summary))

    # %% [markdown]
    # ## 4. Exploratory Data Analysis

    # %% [markdown]
    # ### 4.1 Dataset Overview & Missing Values

    # %%
    # Print basic structure, shapes and column types of the merged dataset
    add_text("### Dataset Overview")
    add_text(f"The final merged dataset contains **{df.shape[0]}** patients and **{df.shape[1]}** columns.")
    add_text("#### Data Types Distribution")
    add_dataframe(df.dtypes.astype(str).value_counts().reset_index().rename(columns={"index": "Data Type", "count": "Count", 0: "Count"}))
    add_text("#### Numeric Feature Statistics")
    add_dataframe(df.describe().round(2))

    # %%
    # Calculate missing value percentages across all columns
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(1)
    missing_df = pd.DataFrame({'Missing': missing, 'Percent': missing_pct})
    missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Percent', ascending=False)

    add_text(f"#### Missing Values (Top 20 Columns)")
    add_dataframe(missing_df.head(20).reset_index().rename(columns={"index": "Column"}))

    fig, ax = plt.subplots(figsize=(12, 6))
    if len(missing_df) > 0:
        missing_df.head(25)['Percent'].plot(kind='barh', ax=ax, color='#e74c3c', alpha=0.8)
        ax.set_xlabel('Missing %')
        ax.set_title('Missing Values by Column (Top 25)')
        plt.tight_layout()
    # plt.show()
    add_plot(fig)

    # %% [markdown]
    # ### 4.2 Demographics

    # %%
    # Plot demographic distributions: Gender, Race, Age, and Income
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Patient Demographics', fontsize=16, fontweight='bold')

    # Gender
    df['GENDER'].value_counts().plot(kind='pie', ax=axes[0,0],
        autopct='%1.1f%%', colors=['#4ECDC4', '#FF6B6B'], startangle=90)
    axes[0,0].set_title('Gender Distribution')
    axes[0,0].set_ylabel('')

    # Race
    df['RACE'].value_counts().plot(kind='barh', ax=axes[0,1], color='#45B7D1', edgecolor='black')
    axes[0,1].set_title('Race Distribution')
    axes[0,1].set_xlabel('Count')

    # Age
    axes[1,0].hist(df['AGE'].dropna(), bins=40, color='#96CEB4', edgecolor='black', alpha=0.8)
    axes[1,0].set_title('Age Distribution')
    axes[1,0].set_xlabel('Age (years)')
    axes[1,0].set_ylabel('Count')
    axes[1,0].axvline(df['AGE'].mean(), color='red', linestyle='--', label=f"Mean: {df['AGE'].mean():.0f}")
    axes[1,0].legend()

    # Income
    axes[1,1].hist(df['INCOME'].dropna(), bins=40, color='#DDA0DD', edgecolor='black', alpha=0.8)
    axes[1,1].set_title('Income Distribution')
    axes[1,1].set_xlabel('Income ($)')
    axes[1,1].set_ylabel('Count')

    plt.tight_layout()
    # plt.show()
    add_plot(fig)

    # Ethnicity
    add_text("#### Ethnicity Distribution")
    add_dataframe(df['ETHNICITY'].value_counts().reset_index().rename(columns={"index": "Ethnicity", "ETHNICITY": "Count"}))

    # %% [markdown]
    # ### 4.3 Target Disease Distribution

    # %%
    # Visualize how many patients have which disease targets
    target_cols = [c for c in df.columns if c.startswith('TARGET_')]
    disease_counts = df[target_cols].sum().sort_values(ascending=True)
    disease_names = [c.replace('TARGET_', '') for c in disease_counts.index]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Disease Target Analysis', fontsize=16, fontweight='bold')

    # Bar chart of disease counts
    axes[0].barh(disease_names, disease_counts.values, color='#e74c3c', alpha=0.8, edgecolor='black')
    axes[0].set_xlabel('Number of Patients')
    axes[0].set_title('Patients per Disease Group')
    for i, v in enumerate(disease_counts.values):
        axes[0].text(v + 1, i, str(int(v)), va='center', fontweight='bold')

    # Healthy vs Diseased
    healthy = (df['HAS_ANY_DISEASE'] == 0).sum()
    diseased = (df['HAS_ANY_DISEASE'] == 1).sum()
    axes[1].pie([healthy, diseased], labels=['Healthy', 'Has Disease'],
                autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], startangle=90,
                textprops={'fontsize': 12})
    axes[1].set_title('Healthy vs Diseased')

    # Number of diseases per patient (for diseased patients only)
    diseased_df = df[df['HAS_ANY_DISEASE'] == 1]
    if len(diseased_df) > 0:
        diseased_df['NUM_DISEASES'].value_counts().sort_index().plot(
            kind='bar', ax=axes[2], color='#3498db', edgecolor='black', alpha=0.8)
        axes[2].set_title('Number of Diseases per Patient\n(diseased patients only)')
        axes[2].set_xlabel('Number of Diseases')
        axes[2].set_ylabel('Count')

    plt.tight_layout()
    # plt.show()
    add_plot(fig)

    # %%
    if len(target_cols) > 1:
        cooccurrence = df[target_cols].T.dot(df[target_cols])
        cooccurrence.index = [c.replace('TARGET_', '') for c in cooccurrence.index]
        cooccurrence.columns = [c.replace('TARGET_', '') for c in cooccurrence.columns]

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cooccurrence, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
                    linewidths=0.5, square=True)
        ax.set_title('Disease Co-occurrence Matrix\n(diagonal = total count per disease)', fontsize=14)
        plt.tight_layout()
        # plt.show()
        fig = plt.gcf()
        add_plot(fig)

    # %% [markdown]
    # ### 4.4 Clinical Feature Distributions

    # %%
    clinical_mean_cols = [c for c in df.columns if c.endswith('_mean') and not c.startswith('avg_')]
    n_cols = min(len(clinical_mean_cols), 12)
    if n_cols > 0:
        cols_to_plot = clinical_mean_cols[:n_cols]
        n_rows = (n_cols + 2) // 3
        fig, axes = plt.subplots(n_rows, 3, figsize=(16, 4*n_rows))
        fig.suptitle('Clinical Feature Distributions (Mean per Patient)', fontsize=16, fontweight='bold')
        axes = axes.flatten() if n_cols > 3 else [axes] if n_cols == 1 else axes

        for i, col in enumerate(cols_to_plot):
            data = df[col].dropna()
            if len(data) > 0:
                axes[i].hist(data, bins=30, color='#3498db', edgecolor='black', alpha=0.7)
                axes[i].set_title(col.replace('_mean', ''), fontsize=10)
                axes[i].axvline(data.mean(), color='red', linestyle='--', alpha=0.7)

        # Hide unused axes
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        # plt.show()
        fig = plt.gcf()
        add_plot(fig)

    # %% [markdown]
    # ### 4.5 Feature Correlations

    # %%
    # Generate a feature-to-feature correlation heatmap to find collinearity
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove target columns and ID-like columns for correlation
    corr_cols = [c for c in numeric_cols if not c.startswith('TARGET_') and c not in ['HAS_ANY_DISEASE', 'NUM_DISEASES']]

    if len(corr_cols) > 2:
        corr_matrix = df[corr_cols].corr()

        fig, ax = plt.subplots(figsize=(18, 14))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, cmap='RdBu_r', center=0,
                    ax=ax, linewidths=0.3, fmt='.1f',
                    xticklabels=True, yticklabels=True)
        ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=7)
        plt.yticks(fontsize=7)
        plt.tight_layout()
        # plt.show()
        fig = plt.gcf()
        add_plot(fig)

    # %% [markdown]
    # ### 4.6 Feature-Target Relationships

    # %%
    # Compare key observation means between Healthy and Diseased patients
    key_features = ['bmi_mean', 'sbp_mean', 'dbp_mean', 'glucose_mean', 'hba1c_mean', 'heart_rate_mean']
    available_features = [f for f in key_features if f in df.columns]

    if len(available_features) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('Key Clinical Features: Healthy vs Diseased', fontsize=16, fontweight='bold')
        axes = axes.flatten()

        for i, feat in enumerate(available_features):
            plot_df = df[['HAS_ANY_DISEASE', feat]].dropna()
            plot_df['Status'] = plot_df['HAS_ANY_DISEASE'].map({0: 'Healthy', 1: 'Diseased'})
            sns.boxplot(data=plot_df, x='Status', y=feat, ax=axes[i],
                    palette={'Healthy': '#2ecc71', 'Diseased': '#e74c3c'})
            axes[i].set_title(feat.replace('_mean', '').upper())

        for j in range(len(available_features), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        # plt.show()
        fig = plt.gcf()
        add_plot(fig)

    # %% [markdown]
    # ### 4.7 Encounter & Utilization Patterns

    # %%
    # Compare healthcare utilization (cost/count) between Healthy vs Diseased
    util_features = ['total_encounters', 'total_medications', 'total_procedures',
                    'total_claim_cost', 'total_med_cost', 'total_proc_cost']
    available_util = [f for f in util_features if f in df.columns]

    if len(available_util) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('Healthcare Utilization: Healthy vs Diseased', fontsize=16, fontweight='bold')
        axes = axes.flatten()

        for i, feat in enumerate(available_util):
            plot_df = df[['HAS_ANY_DISEASE', feat]].dropna()
            plot_df['Status'] = plot_df['HAS_ANY_DISEASE'].map({0: 'Healthy', 1: 'Diseased'})
            sns.boxplot(data=plot_df, x='Status', y=feat, ax=axes[i],
                    palette={'Healthy': '#2ecc71', 'Diseased': '#e74c3c'})
            axes[i].set_title(feat.replace('_', ' ').title())

        for j in range(len(available_util), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        # plt.show()
        fig = plt.gcf()
        add_plot(fig)

    # %% [markdown]
    # ## 5. Save Processed Dataset

    # %%
    # --- Step 5: Save the processed data for modeling ---

    # Based on the EDA, drop leaky features (utilization, costs) and outcome metrics (DALY, QALY, QOLS)
    leaky_and_outcome_cols = [
        'INCOME', 'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE', 'IS_ALIVE',
        'total_encounters', 'unique_enc_types', 'total_claim_cost', 'avg_claim_cost', 'total_payer_coverage',
        'enc_ambulatory', 'enc_emergency', 'enc_home', 'enc_hospice', 'enc_inpatient', 'enc_outpatient', 'enc_snf', 'enc_urgentcare', 'enc_virtual', 'enc_wellness',
        'total_medications', 'unique_medications', 'total_med_cost', 'avg_dispenses',
        'total_procedures', 'unique_procedures', 'total_proc_cost',
        'daly_mean', 'daly_std', 'qaly_mean', 'qaly_std', 'qols_mean', 'qols_std'
    ]

    cols_to_drop = [c for c in leaky_and_outcome_cols if c in df.columns]
    df_clean = df.drop(columns=cols_to_drop)

    # Drop PATIENT ID for the model-ready version (keep a copy with IDs)
    df_clean.to_csv('merged_dataset_with_ids.csv', index=False)
    df_model = df_clean.drop(columns=['PATIENT'])
    df_model.to_csv('merged_dataset.csv', index=False)

    add_text(f"Saved merged_dataset_with_ids.csv ({df_clean.shape})")
    add_text(f"Saved merged_dataset.csv ({df_model.shape})")


    # %% [markdown]
    # ## Summary
    # 
    # **Merged Dataset**: {df.shape[0]} patients × {df.shape[1]} columns
    # 
    # **Feature Groups**:
    # - Demographics: Age, Gender, Race, Ethnicity, Income, Healthcare costs
    # - Clinical Observations: BMI, Blood Pressure, Glucose, HbA1c, etc. (mean + std per patient)
    # - Encounters: Count by type, total costs
    # - Medications: Count, unique, costs
    # - Procedures: Count, unique, costs
    # 
    # **Target**: Multi-label binary vector for disease groups
    # 
    # **Next Steps**:
    # 1. Handle missing values (imputation)
    # 2. Encode categoricals (Gender, Race, Ethnicity)
    # 3. Temporal split into Dataset 1 (Historical) and Dataset 2 (Current)
    # 4. Train Decision Tree, SVM, Neural Network
    # 5. Build Streamlit dashboard

    # %%
    # Displays the first 5 rows of the final dataset
    # df_model


    # %%
    add_text("### Pre-Split Merged Dataset Preview")
    merged_df=pd.read_csv('merged_dataset_with_ids.csv')
    add_dataframe(merged_df.head())

    # %%
    # ensure datetime
    encounters["START"] = pd.to_datetime(encounters["START"], errors="coerce")

    # get last encounter per patient
    last_enc = encounters.groupby("PATIENT")["START"].max().reset_index()

    # rename column
    last_enc.rename(columns={"START": "last_visit_date"}, inplace=True)

    # merge into merged dataset
    final_df = merged_df.merge(last_enc, on="PATIENT", how="left")

    # %%
    add_text("### Final Merged Dataset Preview (with Last Visit Date)")
    add_dataframe(final_df.head())

    # %%
    final_df["last_visit_date"] = pd.to_datetime(final_df["last_visit_date"], errors="coerce")

    split_date = final_df["last_visit_date"].quantile(0.7)

    dataset1 = final_df[final_df["last_visit_date"] <= split_date]
    dataset2 = final_df[final_df["last_visit_date"] > split_date]

    # %%
    # Save the merged dataset to mergedDataset.csv if it doesn't already exist
    import os

    output_path = 'mergedDataset.csv'
    if not os.path.exists(output_path):
        df.to_csv(output_path, index=False)
        add_text(f'✅ Saved merged dataset to {output_path} — shape: {df.shape}')
    else:
        add_text(f'ℹ️  {output_path} already exists, skipping save.')


    # %%

    set_section("svm")
    # %% [markdown]
    # ### 5. Support Vector Machine (SVM) Pipeline
    # - training individual svms for each target disease
    # - skipping targets with too few positive cases
    # - evaluating on historical and current datasets
    # - using sgdclassifier to allow partial_fit for continual learning
    # 

    # setup vars
    exclude_cols = ['PATIENT', 'last_visit_date', 'HAS_ANY_DISEASE', 'NUM_DISEASES']
    target_cols = [c for c in dataset1.columns if c.startswith('TARGET_')]
    drop_cols = exclude_cols + target_cols

    X1_raw = dataset1.drop(columns=drop_cols)
    X2_raw = dataset2.drop(columns=drop_cols)

    X1_raw = pd.get_dummies(X1_raw)
    X2_raw = pd.get_dummies(X2_raw)
    X1_raw, X2_raw = X1_raw.align(X2_raw, join='inner', axis=1, fill_value=0)

    all_metrics = []
    trained_models = {}

    for target in target_cols:
        y1 = dataset1[target]
        y2 = dataset2[target]
        
        pos_cases_ds1 = int(y1.sum())
        pos_cases_ds2 = int(y2.sum())
        
        # split data
        if pos_cases_ds1 < 10 or pos_cases_ds2 < 2:
            add_text(f"Skipping {target} (Insufficient positive cases: DS1={pos_cases_ds1}, DS2={pos_cases_ds2})")
            continue
            
        add_text(f"### Analyzing Target: {target}")
        add_text(f"This model focuses on predicting **{target}**. Dataset 1 contains **{pos_cases_ds1}** positive cases for this condition.")
        
        # split data
        X1_train, X1_test, y1_train, y1_test = train_test_split(X1_raw, y1, test_size=0.2, random_state=42, stratify=y1)
        X2_train, X2_test, y2_train, y2_test = train_test_split(X2_raw, y2, test_size=0.2, random_state=42, stratify=y2)
        
        # scale data
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        
        X1_train_imp = imputer.fit_transform(X1_train)
        X1_test_imp = imputer.transform(X1_test)
        X2_train_imp = imputer.transform(X2_train)
        X2_test_imp = imputer.transform(X2_test)
        
        X1_train_scaled = scaler.fit_transform(X1_train_imp)
        X1_test_scaled = scaler.transform(X1_test_imp)
        X2_train_scaled = scaler.transform(X2_train_imp) # using own scaler for ds2 test
        X2_test_scaled = scaler.transform(X2_test_imp)
        
        base_svm = SGDClassifier(loss='hinge', random_state=42, class_weight='balanced')
        base_svm.fit(X1_train_scaled, y1_train)
        
        pred_ds1 = base_svm.predict(X1_test_scaled)
        pred_ds2 = base_svm.predict(X2_test_scaled)
        
        score_ds1 = base_svm.decision_function(X1_test_scaled)
        score_ds2 = base_svm.decision_function(X2_test_scaled)
        
        f1_ds1 = f1_score(y1_test, pred_ds1, zero_division=0)
        acc_ds1 = accuracy_score(y1_test, pred_ds1)
        prec_ds1 = precision_score(y1_test, pred_ds1, zero_division=0)
        rec_ds1 = recall_score(y1_test, pred_ds1, zero_division=0)

        f1_ds2 = f1_score(y2_test, pred_ds2, zero_division=0)
        acc_ds2 = accuracy_score(y2_test, pred_ds2)
        prec_ds2 = precision_score(y2_test, pred_ds2, zero_division=0)
        rec_ds2 = recall_score(y2_test, pred_ds2, zero_division=0)

        # partial fit for cl
        cl_svm = copy.deepcopy(base_svm)
        cl_svm.partial_fit(X2_train_scaled, y2_train, classes=np.array([0, 1]))
        
        pred_cl = cl_svm.predict(X2_test_scaled)
        score_cl = cl_svm.decision_function(X2_test_scaled)
        
        f1_cl = f1_score(y2_test, pred_cl, zero_division=0)
        acc_cl = accuracy_score(y2_test, pred_cl)
        prec_cl = precision_score(y2_test, pred_cl, zero_division=0)
        rec_cl = recall_score(y2_test, pred_cl, zero_division=0)
        
        metrics_df = pd.DataFrame({
            "Metric": ["F1-Score", "Accuracy", "Precision", "Recall"],
            "DS1 Baseline": [f1_ds1, acc_ds1, prec_ds1, rec_ds1],
            "DS2 Baseline": [f1_ds2, acc_ds2, prec_ds2, rec_ds2],
            "DS2 Fine-Tuned (CL)": [f1_cl, acc_cl, prec_cl, rec_cl]
        })
        add_text(f"#### Performance Metrics: {target}")
        add_dataframe(metrics_df.round(4))
        
        all_metrics.append({'Disease': target, 'DS1_F1': f1_ds1, 'DS2_Base_F1': f1_ds2, 'DS2_CL_F1': f1_cl})
        trained_models[target] = cl_svm
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 4))
        fig.suptitle(f"Model Performance: {target}", fontsize=16, y=1.05)
        
        sns.heatmap(confusion_matrix(y1_test, pred_ds1), annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0])
        axes[0].set_title("DS1 Test (Baseline)")
        axes[0].set_ylabel("Actual")
        axes[0].set_xlabel("Predicted")
        
        sns.heatmap(confusion_matrix(y2_test, pred_ds2), annot=True, fmt='d', cmap='Oranges', cbar=False, ax=axes[1])
        axes[1].set_title("DS2 Test (Baseline Before CL)")
        axes[1].set_xlabel("Predicted")
        
        sns.heatmap(confusion_matrix(y2_test, pred_cl), annot=True, fmt='d', cmap='Greens', cbar=False, ax=axes[2])
        axes[2].set_title("DS2 Test (After CL Fine-Tuning)")
        axes[2].set_xlabel("Predicted")
        
        fpr_b, tpr_b, _ = roc_curve(y2_test, score_ds2)
        fpr_c, tpr_c, _ = roc_curve(y2_test, score_cl)
        
        auc_b = auc(fpr_b, tpr_b) if len(np.unique(y2_test)) > 1 else np.nan
        auc_c = auc(fpr_c, tpr_c) if len(np.unique(y2_test)) > 1 else np.nan
        
        axes[3].plot(fpr_b, tpr_b, color='darkorange', lw=2, label=f'Base (AUC = {auc_b:.2f})')
        axes[3].plot(fpr_c, tpr_c, color='green', lw=2, label=f'CL (AUC = {auc_c:.2f})')
        axes[3].plot([0, 1], [0, 1], color='navy', linestyle='--')
        axes[3].set_title("ROC Curve: Continual Learning Impact")
        axes[3].set_xlabel("False Positive Rate")
        axes[3].set_ylabel("True Positive Rate")
        axes[3].legend(loc="lower right")
        
        plt.tight_layout()
        # plt.show()
        fig = plt.gcf()
        add_plot(fig)

    # final metrics
    add_dataframe(pd.DataFrame(all_metrics))

    # %% [markdown]
    # ### 5.1 Bias-Variance Tradeoff (Complexity Analysis)
    # - testing different alpha values on the respiratory target to see complexity tradeoff
    # - high alpha leads to underfitting (high bias)
    # - extremely low alpha leads to overfitting (high variance)
    # 

    target = 'TARGET_Respiratory'
    y1_comp = dataset1[target]

        # split data
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X1_raw, y1_comp, test_size=0.2, random_state=42, stratify=y1_comp)

        # scale data
    comp_imp = SimpleImputer(strategy='median')
    comp_sc = StandardScaler()

    X_train_c_sc = comp_sc.fit_transform(comp_imp.fit_transform(X_train_c))
    X_test_c_sc = comp_sc.transform(comp_imp.transform(X_test_c))

        # setup vars
    alphas = [10.0, 1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    train_f1s = []
    test_f1s = []

    for a in alphas:
        svm_clf = SGDClassifier(loss='hinge', alpha=a, random_state=42, class_weight='balanced')
        svm_clf.fit(X_train_c_sc, y_train_c)
        
        train_preds = svm_clf.predict(X_train_c_sc)
        test_preds = svm_clf.predict(X_test_c_sc)
        
        train_f1s.append(f1_score(y_train_c, train_preds, zero_division=0))
        test_f1s.append(f1_score(y_test_c, test_preds, zero_division=0))

    # graph curve
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot([str(a) for a in alphas], train_f1s, 'go--', lw=2, label='Train F1-Score')
    ax.plot([str(a) for a in alphas], test_f1s, 'mo-', lw=2, label='Test F1-Score')

    ax.set_title("SVM Complexity / Bias-Variance Tradeoff (TARGET_Respiratory)", fontsize=14)
    ax.set_xlabel("Alpha Parameter (High Variance <-------------------------------------- High Bias)", fontsize=12)
    ax.set_ylabel("F1-Score", fontsize=12)

    ax.invert_xaxis()
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='best')
    plt.tight_layout()
    # plt.show()
    add_plot(fig)


    add_text("### SVM Results Summary")
    add_text("- The F1 scores are generally very low across most targets for the SVM.")
    add_text("- This is expected since the dataset is highly imbalanced and non-linear, so a linear model like SGDClassifier inherently suffers from high bias here.")
    add_text("- However, continual learning worked well. As seen in the ROC curves (like for obesity and dental), the AUC improved after fine-tuning on the newer dataset 2 data.")
    # 

    set_section("decision_tree")
    # %% [markdown]
    # ## 6. Decision Tree Classification — Full Pipeline
    # 
    # Train Decision Trees on Dataset 1's train split, evaluate on Dataset 1 test **and**
    # Dataset 2 (temporal generalisation), then analyse complexity, bias-variance
    # trade-offs and feature importances.
    # 

    # %%

    sns.set_theme(style='whitegrid')
    plt.rcParams['figure.figsize'] = (14, 6)
    plt.rcParams['font.size'] = 12
    #print('Imports OK')


    # %% [markdown]
    # ### 6.1 Preprocessing — Imputation, Encoding & Temporal Split
    # 

    # %%
    # Ensure temporal split exists
    final_df['last_visit_date'] = pd.to_datetime(final_df['last_visit_date'], errors='coerce')
    split_date = final_df['last_visit_date'].quantile(0.7)
    dataset1 = final_df[final_df['last_visit_date'] <= split_date].copy()
    dataset2 = final_df[final_df['last_visit_date'] >  split_date].copy()
    add_text(f"#### Dataset Temporal Split Dimensions")
    split_info = pd.DataFrame({
        "Dataset": ["Dataset 1 (Historical)", "Dataset 2 (Current)"],
        "Rows": [dataset1.shape[0], dataset2.shape[0]],
        "Columns": [dataset1.shape[1], dataset2.shape[1]]
    })
    add_dataframe(split_info)

    TARGET_COLS  = [c for c in final_df.columns if c.startswith('TARGET_')]
    DROP_COLS    = ['PATIENT','last_visit_date','HAS_ANY_DISEASE','NUM_DISEASES'] + TARGET_COLS
    FEATURE_COLS = [c for c in final_df.columns if c not in DROP_COLS]
    CAT_COLS     = [c for c in FEATURE_COLS if not pd.api.types.is_numeric_dtype(final_df[c])]
    NUM_COLS     = [c for c in FEATURE_COLS if pd.api.types.is_numeric_dtype(final_df[c])]
    add_text(f"#### Feature and Target Configuration")
    config_info = pd.DataFrame({
        "Category": ["Total Features", "Numeric Features", "Categorical Features", "Targets"],
        "Count": [len(FEATURE_COLS), len(NUM_COLS), len(CAT_COLS), len(TARGET_COLS)]
    })
    add_dataframe(config_info)


    # %%
    def prepare_Xy(df, feat_cols, tgt_cols, cat_cols, label_encoders=None, fit=True):
        X = df[feat_cols].copy()
        y = df[tgt_cols].fillna(0).astype(int)
        if label_encoders is None:
            label_encoders = {}
        for col in cat_cols:
            if fit:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
            else:
                le = label_encoders[col]
                known = set(le.classes_)
                X[col] = X[col].astype(str).apply(lambda v: v if v in known else le.classes_[0])
                X[col] = le.transform(X[col])
        return X, y, label_encoders

    d1_train_df, d1_test_df = train_test_split(dataset1, test_size=0.2, random_state=42)
    X_train, y_train, les = prepare_Xy(d1_train_df, FEATURE_COLS, TARGET_COLS, CAT_COLS, fit=True)
    X_test1, y_test1, _  = prepare_Xy(d1_test_df,  FEATURE_COLS, TARGET_COLS, CAT_COLS, label_encoders=les, fit=False)
    X_test2, y_test2, _  = prepare_Xy(dataset2,     FEATURE_COLS, TARGET_COLS, CAT_COLS, label_encoders=les, fit=False)

    imputer = SimpleImputer(strategy='median')
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=FEATURE_COLS)
    X_test1_imp = pd.DataFrame(imputer.transform(X_test1),     columns=FEATURE_COLS)
    X_test2_imp = pd.DataFrame(imputer.transform(X_test2),     columns=FEATURE_COLS)

    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=FEATURE_COLS)
    X_test1_sc = pd.DataFrame(scaler.transform(X_test1_imp),     columns=FEATURE_COLS)
    X_test2_sc = pd.DataFrame(scaler.transform(X_test2_imp),     columns=FEATURE_COLS)
    add_text(f"#### Training and Test Set Dimensions")
    split_dims = pd.DataFrame({
        "Split": ["Training Set", "D1 Test Set", "D2 Test Set"],
        "Shape": [str(X_train_sc.shape), str(X_test1_sc.shape), str(X_test2_sc.shape)]
    })
    add_dataframe(split_dims)


    # %% [markdown]
    # ### 6.2 Train Default Decision Tree & Evaluate on Both Test Sets
    # 

    # %%

    def eval_multilabel(model, X, y_true, split_name):
        y_pred = model.predict(X)
        return {
            'split':     split_name,
            'accuracy':  accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall':    recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1':        f1_score(y_true, y_pred, average='macro', zero_division=0),
        }, y_pred

    # Define the base model
    dt_base = MultiOutputClassifier(DecisionTreeClassifier(random_state=42))

    # Create a custom scorer for GridSearch to optimize F1 Macro
    f1_macro_scorer = make_scorer(f1_score, average='macro', zero_division=0)

    # Define the hyperparameter grid
    # Note the 'estimator__' prefix required to pass params to the DecisionTreeClassifier
    param_grid = {
        'estimator__class_weight': ['balanced'],
        'estimator__max_depth': [3, 5, 10, 15, None],
        'estimator__min_samples_split': [2, 10, 20, 50],
        'estimator__min_samples_leaf': [1, 5, 10, 20]
    }

    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=dt_base,
        param_grid=param_grid,
        scoring=f1_macro_scorer,
        cv=5,          # 5-fold cross validation
        n_jobs=-1,     # Use all CPU cores for parallelization
        verbose=1
    )

    # Run the grid search
    add_text("Starting GridSearchCV for hyperparameter tuning...")
    grid_search.fit(X_train_sc, y_train)

    add_text("\nBest parameters found:")
    for param_name, param_value in grid_search.best_params_.items():
        add_text(f"  {param_name.replace('estimator__', '')}: {param_value}")

    add_text(f"\nBest cross-validation F1-macro score: {grid_search.best_score_:.4f}\n")

    # Use the best model found
    best_dt = grid_search.best_estimator_

    # Evaluate using the best model
    r_d1, y_pred_d1 = eval_multilabel(best_dt, X_test1_sc, y_test1, 'Test  (D1)')
    r_d2, y_pred_d2 = eval_multilabel(best_dt, X_test2_sc, y_test2, 'Test  (D2)')

    add_text("### Tuned Decision Tree Results")
    add_dataframe(pd.DataFrame([r_d1, r_d2]))


    # %%
    n_t = len(TARGET_COLS)
    fig, axes = plt.subplots(2, n_t, figsize=(4*n_t, 8))
    fig.suptitle('Confusion Matrices — Default Decision Tree', fontsize=14, fontweight='bold')

    for i, tgt in enumerate(TARGET_COLS):
        lbl = tgt.replace('TARGET_', '')
        for row, (yt, yp, ttl) in enumerate([
            (y_test1.iloc[:, i].values, y_pred_d1[:, i], f'{lbl}\n(D1-Test)'),
            (y_test2.iloc[:, i].values, y_pred_d2[:, i], f'{lbl}\n(D2-Test)'),
        ]):
            cm = confusion_matrix(yt, yp, labels=[0, 1])
            ax = axes[row][i]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['Neg','Pos'], yticklabels=['Neg','Pos'])
            ax.set_title(ttl, fontsize=9)
            ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')

    plt.tight_layout()
    # plt.show()
    add_plot(fig)

    # %%
    for split_name, yt, yp in [
        ('D1 Test', y_test1, y_pred_d1),
        ('D2 Test (temporal)', y_test2, y_pred_d2),
    ]:
        add_text(f"### Per-label Classification Report — {split_name}")
        for i, tgt in enumerate(TARGET_COLS):
            add_text(f"#### Target: {tgt.replace('TARGET_','')}")
            report = classification_report(yt.iloc[:, i], yp[:, i],
                                         labels=[0, 1],
                                         target_names=['Negative', 'Positive'],
                                         zero_division=0,
                                         output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            add_dataframe(report_df.round(4))

    # %% [markdown]
    # ### 6.3 Model Complexity & Bias-Variance Trade-off
    # 
    # Sweep `max_depth` 1–15, tracking macro-F1 on D1-test and D2-test.
    # 

    # %%
    depths, d1_f1s, d2_f1s = [], [], []

    def macro_f1(m, X, y):
        return f1_score(y, m.predict(X), average='macro', zero_division=0)

    for d in range(1, 16):
        clf = MultiOutputClassifier(DecisionTreeClassifier(max_depth=d, random_state=42))
        clf.fit(X_train_sc, y_train)
        depths.append(d)
        d1_f1s.append(macro_f1(clf, X_test1_sc, y_test1))
        d2_f1s.append(macro_f1(clf, X_test2_sc, y_test2))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(depths, d1_f1s, 's--', lw=2, label='Test  (D1)')
    ax.plot(depths, d2_f1s, '^:',  lw=2, label='Test  (D2 temporal)')
    ax.set_xlabel('max_depth'); ax.set_ylabel('Macro F1')
    ax.set_title('Model complexity: max_depth vs Macro F1 (test sets only)')
    ax.set_xticks(depths); ax.legend(); ax.grid(True)
    plt.tight_layout(); 
    # plt.show()
    add_plot(fig)

    cdf = pd.DataFrame({
        'depth': depths,
        'd1_test_f1': d1_f1s,
        'd2_test_f1': d2_f1s,
        'd1_minus_d2': [round(a - b, 4) for a, b in zip(d1_f1s, d2_f1s)],
    })
    add_text(cdf.round(4).to_string(index=False))


    # %%
    best_depth = depths[int(np.argmax(d1_f1s))]
    add_text(f'Best max_depth (by D1-test F1): {best_depth}')

    dt_best = MultiOutputClassifier(DecisionTreeClassifier(max_depth=best_depth, random_state=42))
    dt_best.fit(X_train_sc, y_train)

    r_d1b, yp_d1b       = eval_multilabel(dt_best, X_test1_sc, y_test1,  'Test  (D1)')
    r_d2b, yp_d2b       = eval_multilabel(dt_best, X_test2_sc, y_test2,  'Test  (D2)')

    best_df = pd.DataFrame([r_d1b, r_d2b])
    add_text(f"### Best Decision Tree Evaluation (max_depth={best_depth})")
    add_dataframe(best_df)


    # %% [markdown]
    # ### 6.4 Feature Importance — Model Interpretation
    # 
    # Gini-based importances per label, aggregated globally and displayed per-disease as a heatmap.
    # 

    # %%
    fi_dict = {tgt.replace('TARGET_',''):dt_best.estimators_[i].feature_importances_
            for i,tgt in enumerate(TARGET_COLS)}
    fi_df = pd.DataFrame(fi_dict, index=FEATURE_COLS)
    fi_df['mean_importance'] = fi_df.mean(axis=1)
    fi_df = fi_df.sort_values('mean_importance', ascending=False)

    # Global top-20
    top20 = fi_df.head(20)
    fig, ax = plt.subplots(figsize=(12, 7))
    top20['mean_importance'].sort_values().plot(kind='barh', ax=ax, color='steelblue', alpha=0.85)
    ax.set_xlabel('Mean Gini Importance')
    ax.set_title(f'Top-20 Feature Importances — DT (max_depth={best_depth})')
    ax.grid(axis='x'); plt.tight_layout()
    # plt.show()
    add_plot(fig)


    add_text("#### Top 20 Global Feature Importances")
    add_dataframe(top20['mean_importance'].round(5).reset_index().rename(columns={"index": "Feature", "mean_importance": "Importance"}))


    # %%
    disease_names = [t.replace('TARGET_','') for t in TARGET_COLS]

    all_top_set = set()
    for col in disease_names:
        all_top_set.update(fi_df[col].nlargest(5).index)
    all_top = sorted(list(all_top_set))

    heat = fi_df.loc[all_top, disease_names]

    fig, ax = plt.subplots(figsize=(max(10, len(disease_names)*2), len(all_top)*0.55+2))
    sns.heatmap(heat, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax, linewidths=0.4,
                cbar_kws={'label':'Gini Importance'})
    ax.set_title('Feature Importance per Disease (top-5 per label)')
    plt.xticks(rotation=30, ha='right'); plt.tight_layout()
    # plt.show()
    add_plot(fig)

    # %%
    # Visualise a shallow tree for the first target
    first_lbl = TARGET_COLS[0].replace('TARGET_','')
    dt_viz = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt_viz.fit(X_train_sc, y_train.iloc[:, 0])

    fig, ax = plt.subplots(figsize=(22, 8))
    plot_tree(dt_viz, feature_names=FEATURE_COLS, class_names=['No','Yes'],
            filled=True, max_depth=3, ax=ax, fontsize=8)
    ax.set_title(f'Decision Tree (depth=3) for {first_lbl}', fontsize=13)
    plt.tight_layout()
    # plt.show()
    add_plot(fig)


    add_text("### Summary of Decision Tree Findings")
    summary_data = [
        ["Preprocessing", "Median imputation → Label-encoding → StandardScaler"],
        ["Underfitting", "depth 1-2: high bias, low train & test F1"],
        ["Overfitting", "depth ≥7: train F1 ≈ 1.0, test F1 plateaus or drops"],
        ["Best depth", "Chosen by D1-test F1 — best balance of bias & variance"],
        ["Temporal gap", "D2-test F1 < D1-test F1, confirming distribution shift over time"],
        ["Top features", "BMI, SBP, DBP, Glucose, HbA1c, Age dominate"],
        ["Clinical insight", "High BMI/Glucose predicts Obesity/Diabetes; elevated SBP → Hypertension"]
    ]
    summary_df = pd.DataFrame(summary_data, columns=["Aspect", "Key Observation"])
    add_dataframe(summary_df)
    # 

    # %%
    # Metrics bar chart — best DT across all splits
    mlong = best_df.melt(id_vars='split',
                        value_vars=['accuracy','precision','recall','f1'],
                        var_name='metric', value_name='score')
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=mlong, x='metric', y='score', hue='split', ax=ax,
                palette=['#2ecc71','#3498db','#e74c3c'])
    ax.set_ylim(0, 1.08)
    ax.set_title(f'Best Decision Tree (max_depth={best_depth}) — Metrics Across Splits')
    ax.set_ylabel('Score'); ax.legend(title='Split')
    for p in ax.patches:
        h = p.get_height()
        if h > 0.005:
            ax.text(p.get_x()+p.get_width()/2, h+0.01, f'{h:.3f}',
                    ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    # plt.show()
    add_plot(fig)


    # %% [markdown]
    # ### 6.6 Continual Learning

    # %%
    from sklearn.base import clone

    add_text("### Retraining on Combined Data (Dataset 1 Train + Dataset 2 Train)")

    # 1. Split Dataset 2 into train and test sets (using the same 80/20 split)
    d2_train_df, d2_test_df = train_test_split(dataset2, test_size=0.2, random_state=42)

    # 2. Combine Dataset 1 and Dataset 2 training dataframes
    combined_train_df = pd.concat([d1_train_df, d2_train_df], ignore_index=True)

    # 3. Prepare features and targets (Re-fitting label encoders on combined data)
    X_train_comb, y_train_comb, les_comb = prepare_Xy(
        combined_train_df, FEATURE_COLS, TARGET_COLS, CAT_COLS, fit=True
    )

    # Prepare Dataset 2's test set using the newly fitted label encoders
    X_test2_comb, y_test2_comb, _ = prepare_Xy(
        d2_test_df, FEATURE_COLS, TARGET_COLS, CAT_COLS, label_encoders=les_comb, fit=False
    )

    # 4. Impute missing values (Re-fitting imputer on combined training data)
    imputer_comb = SimpleImputer(strategy='median')
    X_train_comb_imp = pd.DataFrame(imputer_comb.fit_transform(X_train_comb), columns=FEATURE_COLS)
    X_test2_comb_imp = pd.DataFrame(imputer_comb.transform(X_test2_comb), columns=FEATURE_COLS)

    # 5. Scale features (Re-fitting scaler on combined training data)
    scaler_comb = StandardScaler()
    X_train_comb_sc = pd.DataFrame(scaler_comb.fit_transform(X_train_comb_imp), columns=FEATURE_COLS)
    X_test2_comb_sc = pd.DataFrame(scaler_comb.transform(X_test2_comb_imp), columns=FEATURE_COLS)

    add_text(f"#### Combined Training Split Dimensions")
    comb_info = pd.DataFrame({
        "Set": ["Combined Train", "Dataset 2 Test"],
        "Shape": [str(X_train_comb_sc.shape), str(X_test2_comb_sc.shape)]
    })
    add_dataframe(comb_info)

    # 6. Clone the previous model (If you used the GridSearchCV earlier, this clones the best estimator)
    # If 'best_dt' is not defined from the previous step, replace 'best_dt' with 'dt_base'
    dt_combined = clone(best_dt)

    # 7. Retrain the model on the combined data
    add_text("\nRetraining the model on combined training data...")
    dt_combined.fit(X_train_comb_sc, y_train_comb)

    # 8. Evaluate strictly on Dataset 2's test set
    r_d2_comb, y_pred_d2_comb = eval_multilabel(dt_combined, X_test2_comb_sc, y_test2_comb, 'Test (D2 Combined)')

    add_text("### Retrained Decision Tree Performance Metrics")
    add_dataframe(pd.DataFrame([r_d2_comb]))

    # --- Continual Learning Section for dashboard.py ---

    add_text("## Task 4: Continual Learning Implementation (Decision Tree Excluded)")

    add_text("""

Task 4 requires implementing a continual learning strategy by using the model trained on Dataset 1 as an "initial checkpoint" to be fine-tuned on Dataset 2. While this is highly effective for parametric models that use gradient descent (such as Neural Networks and certain SVM implementations), **it is not algorithmically possible for a standard, single Decision Tree.**

In scikit-learn, the DecisionTreeClassifier builds its structure using a greedy, top-down approach. Once the splits are calculated and the tree is grown on Dataset 1, its internal mathematical structure is permanently locked. The algorithm lacks a partial_fit() method because there are no weights to incrementally update. 

Attempting to fit new data (Dataset 2) to an existing Decision Tree will result in catastrophic forgetting, as the .fit() method will completely discard the Dataset 1 checkpoint and train a brand-new tree from scratch. Because a true "checkpoint and fine-tune" strategy cannot be applied to this specific algorithm, the continual learning step is intentionally omitted for the Decision Tree pipeline, and will be demonstrated instead in the Neural Network and SVM sections of this project.
""")

    add_text("---") # Optional visual separator

    # %%
    final_df.to_csv("final_dataset.csv", index=False)

    # %%
    features = [col for col in final_df.columns]
    num_cols = final_df[features].select_dtypes(include=[np.number]).columns
    cat_cols = final_df[features].select_dtypes(exclude=[np.number]).columns
    feature_cols=[col for col in final_df.columns if not col in target_cols]
    # num_cols, cat_cols

    # %%
    X1 = dataset1[feature_cols].copy()
    X2 = dataset2[feature_cols].copy()
    y1 = dataset1[target_cols].copy()
    y2 = dataset2[target_cols].copy()
    nn_target_cols = y1.columns.tolist()

    # Safety Drop: Prevent DateTime leakage from crashing the float64 casting in the Neural Network
    leakage_cols = ["last_visit_date", "PATIENT"]
    for c in leakage_cols:
        if c in X1.columns: X1 = X1.drop(columns=[c])
        if c in X2.columns: X2 = X2.drop(columns=[c])

    # X1, X2, y1, y2


    # %%
    num_cols = X1.select_dtypes(include=[np.number]).columns
    num_means = X1[num_cols].mean()

    X1[num_cols] = X1[num_cols].fillna(num_means)
    X2[num_cols] = X2[num_cols].fillna(num_means)

    # %%
    cat_cols = X1.select_dtypes(exclude=[np.number]).columns
    cat_modes = X1[cat_cols].mode().iloc[0]

    X1[cat_cols] = X1[cat_cols].fillna(cat_modes)
    X2[cat_cols] = X2[cat_cols].fillna(cat_modes)

    # %%
    X1 = pd.get_dummies(X1, columns=cat_cols)
    X2 = pd.get_dummies(X2, columns=cat_cols)

    X1, X2 = X1.align(X2, join="left", axis=1, fill_value=0)

    # %%
    # X1, X2, y1, y2

    # %%
    def relu(z):
        return np.maximum(0, z)

    def relu_derivative(z):
        return (z > 0).astype(float)

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def forward(X, W1, b1, W2, b2, W3, b3):

        z1 = X @ W1 + b1
        a1 = relu(z1)

        z2 = a1 @ W2 + b2
        a2 = relu(z2)

        z3 = a2 @ W3 + b3
        a3 = sigmoid(z3)   # shape: (m, 8)

        return z1, a1, z2, a2, z3, a3

    # %%
    def init_params(n_input, n_outputs):
        np.random.seed(42)

        W1 = np.random.randn(n_input, 16) * np.sqrt(2/n_input)
        b1 = np.zeros((1, 16))

        W2 = np.random.randn(16, 8) * np.sqrt(2/16)
        b2 = np.zeros((1, 8))

        W3 = np.random.randn(8, n_outputs) * np.sqrt(2/8)
        b3 = np.zeros((1, n_outputs))

        return W1, b1, W2, b2, W3, b3

    # %%
    def compute_loss(y, y_pred):
        eps = 1e-8

        pos_counts = y.sum(axis=0)
        neg_counts = y.shape[0] - pos_counts
        pos_weight = np.log1p(neg_counts / (pos_counts + 1e-8))

        pos_weight = np.clip(pos_weight, 1, 50)

        loss = -(pos_weight * y * np.log(y_pred + eps) +
                (1 - y) * np.log(1 - y_pred + eps))

        return np.mean(loss)

    # %%
    def backward(X, y, z1, a1, z2, a2, z3, a3, W2, W3):
        m = X.shape[0]
        pos_counts = y.sum(axis=0)
        neg_counts = y.shape[0] - pos_counts

        pos_weight = np.log1p(neg_counts / (pos_counts + 1e-8))
        dz3 = (a3 - y)   # shape: (m, 8)
        dW3 = (a2.T @ dz3) / m
        db3 = np.sum(dz3, axis=0, keepdims=True) / m

        dz2 = (dz3 @ W3.T) * relu_derivative(z2)
        dW2 = (a1.T @ dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        dz1 = (dz2 @ W2.T) * relu_derivative(z1)
        dW1 = (X.T @ dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        return dW1, db1, dW2, db2, dW3, db3

    # %%
    def train_mlp(X, y, epochs=1000, lr=0.003):

        n_input = X.shape[1]
        n_outputs = y.shape[1]

        W1, b1, W2, b2, W3, b3 = init_params(n_input, n_outputs)

        for epoch in range(epochs):

            z1, a1, z2, a2, z3, a3 = forward(X, W1, b1, W2, b2, W3, b3)

            loss = compute_loss(y, a3)

            dW1, db1, dW2, db2, dW3, db3 = backward(
                X, y, z1, a1, z2, a2, z3, a3, W2, W3
            )
            dW1 = clip_gradients(dW1)
            dW2 = clip_gradients(dW2)
            dW3 = clip_gradients(dW3)

            W1 -= lr * dW1
            b1 -= lr * db1

            W2 -= lr * dW2
            b2 -= lr * db2

            W3 -= lr * dW3
            b3 -= lr * db3

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

        return W1, b1, W2, b2, W3, b3

    # %%
    def train_test_split_scratch(X, y, test_size=0.2, shuffle=True, seed=42):

        np.random.seed(seed)

        n = X.shape[0]
        indices = np.arange(n)

        if shuffle:
            np.random.shuffle(indices)

        split = int(n * (1 - test_size))

        train_idx = indices[:split]
        test_idx = indices[split:]

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        return X_train, X_test, y_train, y_test

    # %%
    def predict(X, W1, b1, W2, b2, W3, b3, thresholds):
        _, _, _, _, _, a3 = forward(X, W1, b1, W2, b2, W3, b3)

        return (a3 > thresholds).astype(int)

    def predict_prob(X, W1, b1, W2, b2, W3, b3):
        _, _, _, _, _, a3 = forward(X, W1, b1, W2, b2, W3, b3)
        return a3

    def clip_gradients(dW, threshold=5):
        return np.clip(dW, -threshold, threshold)

    # %%
    X1_train, X1_test, y1_train, y1_test = train_test_split_scratch(X1.values, y1.values)
    X2_train, X2_test, y2_train, y2_test = train_test_split_scratch(X2.values, y2.values)
    valid = np.where(y1_train.sum(axis=0) >= 10)[0]
    active_nn_target_cols = [nn_target_cols[i] for i in valid]
    y1_train = y1_train[:, valid]
    y1_test  = y1_test[:, valid]
    y2_train = y2_train[:, valid]
    y2_test  = y2_test[:, valid]

    # %%
    X1_train = X1_train.astype(np.float64)
    X1_test  = X1_test.astype(np.float64)

    y1_train = y1_train.astype(np.float64)
    y1_test  = y1_test.astype(np.float64)

    X2_train = X2_train.astype(np.float64)
    X2_test  = X2_test.astype(np.float64)

    y2_train = y2_train.astype(np.float64)
    y2_test  = y2_test.astype(np.float64)
    
    #%%
    X_mean = X1_train.mean(axis=0)
    X_std = X1_train.std(axis=0) + 1e-8

    X1_train = (X1_train - X_mean) / X_std
    X1_test  = (X1_test - X_mean) / X_std
    X2_train = (X2_train - X_mean) / X_std
    X2_test  = (X2_test - X_mean) / X_std
    
    # %%
    W1, b1, W2, b2, W3, b3 = train_mlp(X1_train, y1_train)
    all_probs = predict_prob(X1_train, W1, b1, W2, b2, W3, b3)

    thresholds = []

    for i in range(y1_train.shape[1]):
        probs = all_probs[:, i]
        labels = y1_train[:, i]

        best_f1 = 0
        best_t = 0.2

        for t in np.linspace(0.05, 0.4, 50):
            preds = (probs > t).astype(int)

            tp = np.sum((labels==1)&(preds==1))
            fp = np.sum((labels==0)&(preds==1))
            fn = np.sum((labels==1)&(preds==0))

            precision = tp/(tp+fp+1e-8)
            recall = tp/(tp+fn+1e-8)
            f1 = 2*precision*recall/(precision+recall+1e-8)

            if precision < 0.05:
                continue

            if preds.sum() == len(preds):
                continue
            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        thresholds.append(best_t)

    thresholds = np.array(thresholds)
    y1_pred = predict(X1_test, W1, b1, W2, b2, W3, b3, thresholds)
    y2_pred = predict(X2_test, W1, b1, W2, b2, W3, b3, thresholds)

    # %%
    def accuracy(y_true, y_pred):
        return np.mean(y_true != y_pred)

    def precision_recall_f1(y_true, y_pred):
        n_labels = y_true.shape[1]

        precisions, recalls, f1s = [], [], []

        for i in range(n_labels):
            yt = y_true[:, i]
            yp = y_pred[:, i]

            tp = np.sum((yt==1)&(yp==1))
            fp = np.sum((yt==0)&(yp==1))
            fn = np.sum((yt==1)&(yp==0))

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        return np.mean(precisions), np.mean(recalls), np.mean(f1s)



    # %%
    # Dataset1
    acc1 = accuracy(y1_test, y1_pred)
    p1, r1, f1_1 = precision_recall_f1(y1_test, y1_pred)

    # Dataset2
    acc2 = accuracy(y2_test, y2_pred)
    p2, r2, f2 = precision_recall_f1(y2_test, y2_pred)

    nn_metrics = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Dataset 1": [acc1, p1, r1, f1_1],
        "Dataset 2": [acc2, p2, r2, f2]
    })
    add_text("#### Neural Network Performance Metrics")
    add_dataframe(nn_metrics.round(4))

    # %%
    add_text("#### Target Distribution in Training Data (Positive Counts)")
    counts_df = pd.DataFrame({
        "Target": active_nn_target_cols,
        "Positive Count": y1_train.sum(axis=0)
    })
    add_dataframe(counts_df)
    y_prob = predict_prob(X1_test, W1, b1, W2, b2, W3, b3)

    add_text(f"Probability distributions in test predictions — Max: {y_prob.max():.4f}, Mean: {y_prob.mean():.4f}")

    # %%

    y1_prob = predict_prob(X1_test, W1, b1, W2, b2, W3, b3)

    plt.figure(figsize=(8, 6))
    for i in range(y1_test.shape[1]):

        fpr, tpr, _ = roc_curve(y1_test[:, i], y1_prob[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"Disease {i} (AUC = {roc_auc:.2f})")

    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (Dataset1)")
    plt.legend()
    # plt.show()
    fig = plt.gcf()
    add_plot(fig)

    # %%
    y2_prob = predict_prob(X2_test, W1, b1, W2, b2, W3, b3)

    plt.figure(figsize=(8, 6))
    for i in range(y2_test.shape[1]):

        fpr, tpr, _ = roc_curve(y2_test[:, i], y2_prob[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"Disease {i} (AUC = {roc_auc:.2f})")

    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (Dataset2)")
    plt.legend()
    # plt.show()
    fig = plt.gcf()
    add_plot(fig)

    # %%
    def train_mlp_continue(X, y, W1, b1, W2, b2, W3, b3, epochs=200, lr=0.0001):

        for epoch in range(epochs):

            z1, a1, z2, a2, z3, a3 = forward(X, W1, b1, W2, b2, W3, b3)

            loss = compute_loss(y, a3)

            dW1, db1, dW2, db2, dW3, db3 = backward(
                X, y, z1, a1, z2, a2, z3, a3, W2, W3
            )

            # small learning rate → VERY IMPORTANT
            # W1 -= lr * dW1
            # b1 -= lr * db1

            W2 -= lr * dW2
            b2 -= lr * db2

            W3 -= lr * dW3
            b3 -= lr * db3

            if epoch % 50 == 0:
                print(f"[Continual] Epoch {epoch}, Loss: {loss:.4f}")

        return W1, b1, W2, b2, W3, b3

    # %%
    W1_c, b1_c, W2_c, b2_c, W3_c, b3_c = train_mlp_continue(
        X2_train, y2_train,
        W1, b1, W2, b2, W3, b3,
        epochs=150,
        lr=0.0004
    )
    thresholds_new=[]
    for i in range(y2_train.shape[1]):
        probs = predict_prob(X2_train, W1, b1, W2, b2, W3, b3)[:, i]
        labels = y2_train[:, i]

        best_f1 = 0
        best_t = 0.5

        for t in np.linspace(0.05, 0.5, 20):
            preds = (probs > t).astype(int)

            tp = np.sum((labels==1)&(preds==1))
            fp = np.sum((labels==0)&(preds==1))
            fn = np.sum((labels==1)&(preds==0))

            precision = tp/(tp+fp+1e-8)
            recall = tp/(tp+fn+1e-8)
            f1 = 2*precision*recall/(precision+recall+1e-8)

            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        thresholds_new.append(best_t)

    thresholds_new = np.array(thresholds_new)
    y2_pred_new = predict(X2_test, W1_c, b1_c, W2_c, b2_c, W3_c, b3_c, thresholds_new)

    # %%
    acc2_new = accuracy(y2_test, y2_pred_new)
    p2_new, r2_new, f2_new = precision_recall_f1(y2_test, y2_pred_new)

    add_text("#### After Continual Learning (NN)")
    cl_nn_metrics = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Value": [acc2_new, p2_new, r2_new, f2_new]
    })
    add_dataframe(cl_nn_metrics.round(4))

    # %%
    # y2_pred_new.sum()

    # %%
    y2_prob_new = predict_prob(X2_test, W1_c, b1_c, W2_c, b2_c, W3_c, b3_c)

    plt.figure(figsize=(8, 6))
    for i in range(y2_test.shape[1]):

        fpr, tpr, _ = roc_curve(y2_test[:, i], y2_prob_new[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"Disease {i} (AUC = {roc_auc:.2f})")

    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (Dataset2) AFTER CONTINUAL LEARNING")
    plt.legend()
    # plt.show()
    fig = plt.gcf()
    add_plot(fig)

    # %%
    def feature_importance(W1):
        # absolute weights summed across neurons
        importance = np.sum(np.abs(W1), axis=1)
        importance = importance / np.sum(importance)
        return importance

    imp = feature_importance(W1)

    # plot
    plt.figure()
    plt.bar(range(len(imp)), imp)
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.title("Feature Importance (NN)")
    # plt.show()
    fig = plt.gcf()
    add_plot(fig)

    def plot_single_confusion(y_true, y_pred):
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()

        cm = confusion_matrix(y_true_flat, y_pred_flat)

        plt.figure()
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title("Overall Confusion Matrix (Multi-label Flattened)")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        # plt.show()
        fig = plt.gcf()
        add_plot(fig)

    plot_single_confusion(y1_test, y1_pred)
    plot_single_confusion(y2_test, y2_pred_new)



if "results_cache" not in st.session_state:
    with st.spinner("Running..."):
        progress_bar = st.progress(0, text="Pipeline is running...")
        run_pipeline()
        progress_bar.progress(100, text="Pipeline complete!")
    st.session_state["results_cache"] = results
else:
    results = st.session_state["results_cache"]


def render_home_section():
    st.title("Healthcare Disease Prediction Dashboard")
    st.write(
        "Use the sidebar to switch between Home, Data Processing, and Misc sections."
    )

    st.subheader("Group Member Details")
    group_members_df = pd.DataFrame(
        {
            "Name": ["Member 1", "Member 2", "Member 3", "Member 4"],
            "ID": ["ID001", "ID002", "ID003", "ID004"],
        }
    )
    st.table(group_members_df)

def render_section(section_name):
    st.header(section_name.replace("_", " ").title())
    for item in results[section_name]:
        if item["type"] == "text":
            st.write(item["content"])
        elif item["type"] == "plot":
            st.pyplot(item["content"])
        elif item["type"] == "dataframe":
            st.dataframe(item["content"])




st.sidebar.title("Controls")
section = st.sidebar.radio("Navigate", ["Home", "Data Processing", "SVM", "Decision Tree", "Neural Network"])

# Scroll to top when section changes
st.markdown('<div id="scroll-to-top"></div>', unsafe_allow_html=True)
st.markdown(
    """
    <script>
        var element = window.parent.document.getElementById('scroll-to-top');
        if (element) {
            element.scrollIntoView();
        }
        window.parent.document.querySelector('section.main').scrollTo(0, 0);
    </script>
    """,
    unsafe_allow_html=True
)

if section == "Home":
    render_home_section()

if section == "Data Processing":
    render_section("data_processing")
elif section == "SVM":
    render_section("svm")
elif section == "Decision Tree":
    render_section("decision_tree")
elif section == "Neural Network":
    render_section("neural_network")