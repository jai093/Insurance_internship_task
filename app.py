import pickle
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# load model and scaler
model = pickle.load(open('model_gb.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.set_page_config(page_title="Insurance Premium App", layout="wide")

# sidebar navigation
page = st.sidebar.selectbox("Navigate", ["Prediction", "Analytics & Insights"])

# ─── PAGE 1: PREDICTION ───────────────────────────────────────────────────────
if page == "Prediction":
    st.title("Insurance Premium Price Prediction")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input('Age', min_value=1, max_value=100, value=25)
        gender = st.selectbox('Gender', ('male', 'female'))
        bmi = st.number_input('BMI', min_value=10.0, max_value=100.0, value=30.0)
    with col2:
        smoker = st.selectbox('Smoker', ('yes', 'no'))
        children = st.number_input('Number of Children', min_value=0, max_value=10, value=2)
        region = st.selectbox('Region', ('southwest', 'southeast', 'northwest', 'northeast'))

    # encoding
    Smoker = 1 if smoker == 'yes' else 0
    sex_female = 1 if gender == 'female' else 0
    sex_male = 1 if gender == 'male' else 0
    region_dict = {'southwest': 0, 'northwest': 1, 'northeast': 2, 'southeast': 3}
    Region = region_dict[region]

    input_features = pd.DataFrame({
        'age': [age], 'bmi': [bmi], 'children': [children],
        'Smoker': [Smoker], 'sex_female': [sex_female],
        'sex_male': [sex_male], 'Region': [Region]
    })
    input_features[['age', 'bmi']] = scaler.transform(input_features[['age', 'bmi']])

    if st.button('Predict Premium'):
        predictions = model.predict(input_features)
        output = round(np.exp(predictions[0]), 2)
        st.success(f"Estimated Premium: ${output:,.2f}")


# ─── PAGE 2: ANALYTICS & INSIGHTS ────────────────────────────────────────────
elif page == "Analytics & Insights":
    st.title("Analytics & Insights")

    df = pd.read_csv('insurance.csv')

    # ── Feature Importance ──────────────────────────────────────────────────
    st.header("Feature Importance")
    feature_names = ['age', 'bmi', 'children', 'Smoker', 'sex_female', 'sex_male', 'Region']
    importances = model.feature_importances_
    fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    fi_df = fi_df.sort_values('Importance')

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(fi_df)))
    ax.barh(fi_df['Feature'], fi_df['Importance'], color=colors)
    ax.set_xlabel('Importance Score')
    ax.set_title('Gradient Boosting Feature Importance')
    ax.spines[['top', 'right']].set_visible(False)
    st.pyplot(fig)
    plt.close()

    st.divider()

    # ── EDA Section ─────────────────────────────────────────────────────────
    st.header("Exploratory Data Analysis (EDA)")

    eda_source = st.radio("Data source", ["Use insurance.csv", "Upload a CSV file"], horizontal=True)

    if eda_source == "Upload a CSV file":
        uploaded = st.file_uploader("Upload a CSV file for visualization", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            st.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")
        else:
            st.info("Upload a CSV file to explore visualizations.")
            st.stop()

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # summary stats
    st.subheader("Summary Statistics")
    st.dataframe(df.describe(), use_container_width=True)

    st.divider()

    # charges distribution
    if 'charges' in df.columns:
        st.subheader("Charges Distribution")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(df['charges'], bins=40, color='steelblue', edgecolor='white')
        axes[0].set_title('Charges Distribution')
        axes[0].set_xlabel('Charges ($)')
        axes[0].spines[['top', 'right']].set_visible(False)

        axes[1].hist(np.log(df['charges']), bins=40, color='teal', edgecolor='white')
        axes[1].set_title('Log(Charges) Distribution')
        axes[1].set_xlabel('Log Charges')
        axes[1].spines[['top', 'right']].set_visible(False)
        st.pyplot(fig)
        plt.close()

    st.divider()

    # smoker vs charges
    if {'smoker', 'charges'}.issubset(df.columns):
        st.subheader("Smoker vs Charges")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for i, (grp, color) in enumerate(zip(['no', 'yes'], ['steelblue', 'tomato'])):
            subset = df[df['smoker'] == grp]['charges']
            axes[0].hist(subset, bins=30, alpha=0.7, label=grp, color=color, edgecolor='white')
        axes[0].set_title('Charges by Smoker Status')
        axes[0].set_xlabel('Charges ($)')
        axes[0].legend(title='Smoker')
        axes[0].spines[['top', 'right']].set_visible(False)

        avg = df.groupby('smoker')['charges'].mean().reset_index()
        axes[1].bar(avg['smoker'], avg['charges'], color=['steelblue', 'tomato'], edgecolor='white')
        axes[1].set_title('Average Charges by Smoker Status')
        axes[1].set_ylabel('Avg Charges ($)')
        axes[1].spines[['top', 'right']].set_visible(False)
        st.pyplot(fig)
        plt.close()

    st.divider()

    # age vs charges
    if {'age', 'charges'}.issubset(df.columns):
        st.subheader("Age vs Charges")
        fig, ax = plt.subplots(figsize=(8, 4))
        hue_col = 'smoker' if 'smoker' in df.columns else None
        if hue_col:
            for grp, color in zip(['no', 'yes'], ['steelblue', 'tomato']):
                sub = df[df['smoker'] == grp]
                ax.scatter(sub['age'], sub['charges'], alpha=0.4, s=15, label=grp, color=color)
            ax.legend(title='Smoker')
        else:
            ax.scatter(df['age'], df['charges'], alpha=0.4, s=15, color='steelblue')
        ax.set_xlabel('Age')
        ax.set_ylabel('Charges ($)')
        ax.set_title('Age vs Charges')
        ax.spines[['top', 'right']].set_visible(False)
        st.pyplot(fig)
        plt.close()

    st.divider()

    # BMI vs charges
    if {'bmi', 'charges'}.issubset(df.columns):
        st.subheader("BMI vs Charges")
        fig, ax = plt.subplots(figsize=(8, 4))
        hue_col = 'smoker' if 'smoker' in df.columns else None
        if hue_col:
            for grp, color in zip(['no', 'yes'], ['steelblue', 'tomato']):
                sub = df[df['smoker'] == grp]
                ax.scatter(sub['bmi'], sub['charges'], alpha=0.4, s=15, label=grp, color=color)
            ax.legend(title='Smoker')
        else:
            ax.scatter(df['bmi'], df['charges'], alpha=0.4, s=15, color='teal')
        ax.set_xlabel('BMI')
        ax.set_ylabel('Charges ($)')
        ax.set_title('BMI vs Charges')
        ax.spines[['top', 'right']].set_visible(False)
        st.pyplot(fig)
        plt.close()

    st.divider()

    # region breakdown
    if {'region', 'charges'}.issubset(df.columns):
        st.subheader("Average Charges by Region")
        avg_region = df.groupby('region')['charges'].mean().sort_values()
        fig, ax = plt.subplots(figsize=(7, 3))
        colors = plt.cm.viridis(np.linspace(0.3, 0.8, len(avg_region)))
        ax.barh(avg_region.index, avg_region.values, color=colors, edgecolor='white')
        ax.set_xlabel('Avg Charges ($)')
        ax.set_title('Average Charges by Region')
        ax.spines[['top', 'right']].set_visible(False)
        st.pyplot(fig)
        plt.close()

    st.divider()

    # correlation heatmap
    st.subheader("Correlation Heatmap")
    num_df = df.select_dtypes(include=np.number)
    if not num_df.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(num_df.corr(), annot=True, fmt='.2f', cmap='coolwarm',
                    linewidths=0.5, ax=ax)
        ax.set_title('Feature Correlation')
        st.pyplot(fig)
        plt.close()
