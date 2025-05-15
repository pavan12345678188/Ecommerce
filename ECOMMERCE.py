import joblib
import streamlit as st # type: ignore
import pandas as pd # type: ignore
import plotly.express as px # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import classification_report, accuracy_score # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore

# Load data
df = pd.read_csv("E Commerce Dataset.csv")
# Load model and dataset
model = joblib.load('ecomdata(1).pkl')
df = joblib.load('df1().pkl')

df.drop(columns=["Unnamed: 0"], inplace=True)
df.drop(columns=["CustomerID"], inplace=True)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})  # Convert Churn to binary
df["CityTier"] = df["CityTier"].map({1: "Tier 1", 2: "Tier 2", 3: "Tier 3"})  # Map CityTier to string
df["CashbackAmount"] = df["CashbackAmount"].replace({"\$": "", ",": ""}, regex=True).astype(float)  # Clean CashbackAmount
df["CashbackAmount"] = df["CashbackAmount"].fillna(0)  # Fill NaN values with 0
df["OrderCount"] = df["OrderCount"].fillna(0)  # Fill NaN values with 0
df["CashbackAmount"] = df["CashbackAmount"].astype(float)  # Ensure CashbackAmount is float
df["OrderCount"] = df["OrderCount"].astype(int)  # Ensure OrderCount is int
df["CashbackAmount"] = df["CashbackAmount"].astype(float)  # Ensure CashbackAmount is float
df["Churn"] = df["Churn"].astype(int)  # Ensure Churn is int
df["CityTier"] = df["CityTier"].astype("category")  # Ensure CityTier is categorical

st.set_page_config(page_title="E-Commerce Dashboard", layout="wide")
st.title("🛍️ E-Commerce Dashboard & Churn Prediction")

# Sidebar for navigation
page = st.sidebar.radio("Go to", ["📊 Dashboard", "🔮 Churn Prediction"])

# ---------- PAGE 1: DASHBOARD ----------
if page == "📊 Dashboard":
    st.header("📈 Business Overview")

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", df.shape[0])
    col2.metric("Total Orders", int(df["OrderCount"].sum()))
    col3.metric("Avg Cashback", f"${df['CashbackAmount'].mean():.2f}")

    # Pie chart: Churn Distribution
    churn_fig = px.pie(df, names="Churn", title="Churn Distribution")
    st.plotly_chart(churn_fig, use_container_width=True)

    # Bar: Order Count by CityTier
    city_fig = px.bar(df, x="CityTier", y="OrderCount", color="CityTier", title="Orders by City Tier")
    st.plotly_chart(city_fig, use_container_width=True)

    # Box: Revenue vs Churn
    revenue_fig = px.box(df, x="Churn", y="CashbackAmount", color="Churn", title="Cashback by Churn")
    st.plotly_chart(revenue_fig, use_container_width=True)

    st.subheader("📄 Raw Data")
    st.dataframe(df)

# ---------- PAGE 2: CHURN PREDICTION ----------
elif page == "🔮 Churn Prediction":
    st.header("📌 Predict Customer Churn")

    # Feature selection
    default_features = [col for col in df.columns if col not in ["Churn"]]
    selected_features = st.multiselect("Select Features for Prediction", options=default_features, default=default_features)

    if len(selected_features) > 0:
        X = df[selected_features]
        y = df["Churn"]

        # Preprocessing
        X = pd.get_dummies(X, drop_first=True)  # Encode categoricals
        X.fillna(0, inplace=True)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Output results
        acc = accuracy_score(y_test, y_pred)
        st.success(f"✅ Model Accuracy: {acc:.2f}")

        st.text("📋 Classification Report")
        st.code(classification_report(y_test, y_pred))

        # Predict for all
        df["Churn_Predicted"] = model.predict(X_scaled)
        st.subheader("🔍 Predicted Churn on Full Dataset")
        st.dataframe(df[["Churn", "Churn_Predicted"] + selected_features])

        # Download predictions
        st.download_button("📥 Download Predictions as CSV", df.to_csv(index=False), "churn_predictions.csv", "text/csv")

    else:
        st.warning("Please select at least one feature to continue.")

