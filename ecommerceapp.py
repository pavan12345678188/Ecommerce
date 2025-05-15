import streamlit as st # type: ignore
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("E Commerce.csv")
df.drop(columns=["Unnamed: 0"], inplace=True)
df.drop(columns=['CustomerID'], inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df['CityTier'] = df['CityTier'].map({'Tier 1': 1, 'Tier 2': 2, 'Tier 3': 3})
df['OrderCount'] = df['OrderCount'].astype(int)
df['CashbackAmount'] = df['CashbackAmount'].astype(float)
df['CashbackAmount'] = df['CashbackAmount'].replace(0, df['CashbackAmount'].mean())

st.set_page_config(page_title="E-Commerce Dashboard", layout="wide")
st.title("🛍️ E-Commerce Dashboard & Churn Prediction")

# Sidebar
page = st.sidebar.radio("Navigate", ["📊 Dashboard", "🔮 Churn Prediction"])

# ---------- PAGE 1: DASHBOARD ----------
if page == "📊 Dashboard":
    st.header("📈 Business Metrics")

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", df.shape[0])
    col2.metric("Total Orders", int(df["OrderCount"].sum()))
    col3.metric("Avg Cashback", f"${df['CashbackAmount'].mean():.2f}")

    # Pie chart using matplotlib
    st.subheader("Churn Distribution")
    churn_counts = df['Churn'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)

    # Barplot: Orders by CityTier
    st.subheader("Orders by City Tier")
    fig2, ax2 = plt.subplots()
    sns.barplot(x='CityTier', y='OrderCount', data=df, ax=ax2)
    st.pyplot(fig2)

    # Boxplot: Cashback by Churn
    st.subheader("Cashback Distribution by Churn")
    fig3, ax3 = plt.subplots()
    sns.boxplot(x='Churn', y='CashbackAmount', data=df, ax=ax3)
    st.pyplot(fig3)

    st.subheader("📄 Raw Dataset")
    st.dataframe(df)

# ---------- PAGE 2: CHURN PREDICTION ----------
elif page == "🔮 Churn Prediction":
    st.header("📌 Predict Customer Churn")

    # Feature selector
    default_features = [col for col in df.columns if col != "Churn"]
    selected_features = st.multiselect("Select Features to Train Model", options=default_features, default=default_features)

    if selected_features:
        X = df[selected_features]
        y = df["Churn"]

        # Encode + clean
        X = pd.get_dummies(X, drop_first=True)
        X.fillna(0, inplace=True)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

        # Model training
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Output
        acc = accuracy_score(y_test, y_pred)
        st.success(f"✅ Model Accuracy: {acc:.2f}")

        st.text("📋 Classification Report")
        st.code(classification_report(y_test, y_pred))

        # Predict full data
        df["Churn_Predicted"] = model.predict(X_scaled)
        st.subheader("🔍 Predicted Churn on All Customers")
        st.dataframe(df[["Churn", "Churn_Predicted"] + selected_features])

        st.download_button("📥 Download Predictions as CSV", df.to_csv(index=False), "churn_predictions.csv", "text/csv")
    else:
        st.warning("Please select at least one feature to proceed.")
