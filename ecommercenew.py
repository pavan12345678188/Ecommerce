import joblib
import streamlit as st # type: ignore
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("E Commerce Dataset.csv")

# Clean and prepare
df.drop(columns=["Unnamed: 0", "CustomerID"], inplace=True, errors='ignore')
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)
df["CityTier"] = df["CityTier"].map({1: "Tier 1", 2: "Tier 2", 3: "Tier 3"}).astype("category")
df["CashbackAmount"] = df["CashbackAmount"].replace({"\\$": "", ",": ""}, regex=True).astype(float)
df["CashbackAmount"].fillna(0, inplace=True)
df["OrderCount"].fillna(0, inplace=True)
df["OrderCount"] = df["OrderCount"].astype(int)

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
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Model training
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.success(f"✅ Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        st.text("📋 Classification Report")
        st.code(classification_report(y_test, y_pred))

        df["Churn_Predicted"] = model.predict(X_scaled)
        st.subheader("🔍 Predicted Churn on Full Dataset")
        st.dataframe(df[["Churn", "Churn_Predicted"] + selected_features])

        st.download_button("📥 Download Predictions", df.to_csv(index=False), "churn_predictions.csv", "text/csv")

    else:
        st.warning("Please select at least one feature to continue.")
