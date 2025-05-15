import joblib
import streamlit as st # type: ignore
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("E Commerce Dataset.csv")

# Clean and prepare
df.drop(columns=["Unnamed: 0", "CustomerID"], inplace=True, errors='ignore')
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
df["CityTier"] = df["CityTier"].map({1: "Tier 1", 2: "Tier 2", 3: "Tier 3"}).astype("category")
df["CashbackAmount"] = df["CashbackAmount"].replace({"\\$": "", ",": ""}, regex=True)
df["CashbackAmount"] = pd.to_numeric(df["CashbackAmount"], errors='coerce').fillna(0).astype(float)
df["OrderCount"] = pd.to_numeric(df["OrderCount"], errors='coerce').fillna(0).astype(int)

# UI Setup
st.set_page_config(page_title="E-Commerce Dashboard", layout="wide")
st.title("🛍️ E-Commerce Dashboard & Churn Prediction")

# Sidebar Navigation
page = st.sidebar.radio("Go to", ["📊 Dashboard", "🔮 Churn Prediction"])

# ---------- DASHBOARD ----------
if page == "📊 Dashboard":
    st.header("📈 Business Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", df.shape[0])
    col2.metric("Total Orders", int(df["OrderCount"].sum()))
    col3.metric("Avg Cashback", f"${df['CashbackAmount'].mean():.2f}")

    # Matplotlib + Seaborn visualizations
    st.subheader("Churn Distribution")
    churn_counts = df['Churn'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(churn_counts, labels=['No', 'Yes'], autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)

    st.subheader("Orders by City Tier")
    fig2, ax2 = plt.subplots()
    sns.barplot(data=df, x="CityTier", y="OrderCount", ax=ax2)
    st.pyplot(fig2)

    st.subheader("Cashback by Churn")
    fig3, ax3 = plt.subplots()
    sns.boxplot(data=df, x="Churn", y="CashbackAmount", ax=ax3)
    st.pyplot(fig3)

    st.subheader("📄 Raw Dataset")
    st.dataframe(df)

# ---------- CHURN PREDICTION ----------
elif page == "🔮 Churn Prediction":
    st.header("📌 Predict Customer Churn")

    # Select Features
    default_features = [col for col in df.columns if col not in ["Churn"]]
    selected_features = st.multiselect("Select Features for Prediction", options=default_features, default=default_features)

    if selected_features:
        X = df[selected_features]
        y = df["Churn"]

        X = pd.get_dummies(X, drop_first=True)
        X.fillna(0, inplace=True)

        # Load saved model and scaler
        model = joblib.load("churn_model.pkl")
        scaler = joblib.load("churn_scaler.pkl")
        X_scaled = scaler.transform(X)

        # Predict using loaded model
        y_pred = model.predict(X_scaled)
        accuracy = accuracy_score(y, y_pred)

        st.success(f"✅ Model Accuracy (on full data): {accuracy:.2f}")
        st.text("📋 Classification Report")
        st.code(classification_report(y, y_pred))

        df["Churn_Predicted"] = y_pred
        st.subheader("🔍 Predicted Churn on Full Dataset")
        st.dataframe(df[["Churn", "Churn_Predicted"] + selected_features])

        st.download_button("📥 Download Predictions", df.to_csv(index=False), "churn_predictions.csv", "text/csv")

    else:
        st.warning("Please select at least one feature to continue.")
