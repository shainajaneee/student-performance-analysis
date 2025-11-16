import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Streamlit Page Config ---
st.set_page_config(page_title="Student Performance Dashboard", layout="wide")
st.title("Student Performance Analysis Dashboard")

# --- Upload multiple CSV files ---
uploaded_files = st.file_uploader(
    "Upload your student performance CSV files (Math and Portuguese)", 
    type=["csv"], 
    accept_multiple_files=True
)

if uploaded_files:
    # Read and merge all uploaded CSVs
    df_list = []
    for file in uploaded_files:
        df_list.append(pd.read_csv(file, sep=';'))  # use sep=';' if needed
    df = pd.concat(df_list, ignore_index=True)
    df.columns = df.columns.str.strip()  # remove any extra spaces

    # Show columns
    st.subheader("Columns in the dataset")
    st.write(df.columns.tolist())

    # --- Create Pass/Fail column based on G3 ---
    if 'G3' in df.columns:
        df['Pass/Fail'] = df['G3'].apply(lambda x: 'pass' if x >= 10 else 'fail')

        # --- Pass vs Fail Distribution ---
        st.subheader("Pass vs Fail Distribution")
        counts = df['Pass/Fail'].value_counts()
        fig1, ax1 = plt.subplots()
        sns.barplot(x=counts.index, y=counts.values, palette='pastel', ax=ax1)
        ax1.set_ylabel("Number of Students")
        st.pyplot(fig1)

        # --- Grade Distribution ---
        st.subheader("Grade Distribution (G1, G2, G3)")
        fig2, ax2 = plt.subplots(figsize=(10,6))
        for col, color in zip(['G1', 'G2', 'G3'], ['skyblue', 'orange', 'green']):
            if col in df.columns:
                sns.histplot(df[col], kde=True, bins=10, alpha=0.6, label=col, color=color, ax=ax2)
        ax2.set_xlabel("Grades")
        ax2.set_ylabel("Number of Students")
        ax2.legend()
        st.pyplot(fig2)

        # --- Correlation Heatmap ---
        st.subheader("Correlation Heatmap")
        numeric_cols = df.select_dtypes(include='number')
        fig3, ax3 = plt.subplots(figsize=(12,8))
        sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', ax=ax3)
        st.pyplot(fig3)

    else:
        st.warning("Column 'G3' not found. Cannot create Pass/Fail or grade charts.")

    st.success("Dashboard successfully generated!")
else:
    st.info("Please upload at least one CSV file to generate the dashboard.")
