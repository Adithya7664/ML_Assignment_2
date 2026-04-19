import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Healthcare ML Dashboard", layout="wide")

# -----------------------
# Sidebar
# -----------------------
st.sidebar.title("Controls")

disease = st.sidebar.selectbox(
    "Select Disease",
    ["Diabetes", "Hypertension", "Heart Disease"]
)

model = st.sidebar.selectbox(
    "Select Model",
    ["Decision Tree", "SVM", "Neural Network"]
)

# -----------------------
# Title
# -----------------------
st.title("🧠 Healthcare Disease Prediction Dashboard")

st.markdown(f"### Selected Disease: **{disease}**")
st.markdown(f"### Selected Model: **{model}**")

# -----------------------
# Placeholder Metrics
# -----------------------
st.subheader("📊 Model Performance")

# dummy values
accuracy = np.random.uniform(0.75, 0.95)
precision = np.random.uniform(0.7, 0.9)
recall = np.random.uniform(0.7, 0.9)
f1 = 2 * (precision * recall) / (precision + recall)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{accuracy:.2f}")
col2.metric("Precision", f"{precision:.2f}")
col3.metric("Recall", f"{recall:.2f}")
col4.metric("F1 Score", f"{f1:.2f}")

# -----------------------
# Model Comparison Chart
# -----------------------
st.subheader("📈 Model Comparison")

comparison_data = pd.DataFrame({
    "Model": ["Decision Tree", "SVM", "Neural Network"],
    "Accuracy": np.random.uniform(0.7, 0.95, 3)
})

st.bar_chart(comparison_data.set_index("Model"))

# -----------------------
# Feature Importance (Placeholder)
# -----------------------
st.subheader("🔍 Feature Importance")

features = ["Age", "Gender", "Encounters", "Glucose", "BMI", "BP"]
importance = np.random.uniform(0.1, 1, len(features))

feat_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

st.bar_chart(feat_df.set_index("Feature"))

# -----------------------
# Data Insights (Placeholder)
# -----------------------
st.subheader("📌 Insights")

st.write("""
- Patients with higher glucose levels show higher disease probability  
- Increased encounter frequency correlates with chronic conditions  
- BMI and blood pressure are strong indicators  
""")

# -----------------------
# Raw Data Section (Optional)
# -----------------------
st.subheader("📂 Sample Data")

dummy_data = pd.DataFrame(np.random.randn(10, 6), columns=features)
st.dataframe(dummy_data)

st.subheader("📊 Cross-Dataset Performance")

cross_data = pd.DataFrame({
    "Scenario": ["Train D1 → Test D1", "Train D1 → Test D2", "After Fine-Tuning"],
    "Accuracy": np.random.uniform(0.6, 0.95, 3)
})

st.bar_chart(cross_data.set_index("Scenario"))