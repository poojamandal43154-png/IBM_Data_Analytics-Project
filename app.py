import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Analytics & AI",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("📊 Customer Churn Analytics & AI")
st.write(
    "A Data Analytics and Machine Learning application "
    "for analyzing customer behavior and predicting churn."
)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("customer_churn.csv")

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Fill missing values
    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].median()
    )

    return df


df = load_data()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Dashboard",
        "Exploratory Data Analysis",
        "Churn Prediction",
        "AI-Powered Insights"
    ]
)

# ---------------------------------------------------
# DASHBOARD
# ---------------------------------------------------

if page == "Dashboard":

    st.header("📌 Customer Overview")

    total_customers = len(df)
    churned = (df["Churn"] == "Yes").sum()
    retained = (df["Churn"] == "No").sum()
    churn_rate = churned / total_customers * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Customers",
        total_customers
    )

    col2.metric(
        "Churned Customers",
        churned
    )

    col3.metric(
        "Retained Customers",
        retained
    )

    col4.metric(
        "Churn Rate",
        f"{churn_rate:.2f}%"
    )

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Number of Rows:", df.shape[0])

    with col2:
        st.write("Number of Columns:", df.shape[1])

# ---------------------------------------------------
# EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------

elif page == "Exploratory Data Analysis":

    st.header("📈 Exploratory Data Analysis")

    # Churn distribution
    st.subheader("Customer Churn Distribution")

    fig, ax = plt.subplots()

    sns.countplot(
        data=df,
        x="Churn",
        ax=ax
    )

    ax.set_xlabel("Churn")
    ax.set_ylabel("Number of Customers")

    st.pyplot(fig)

    # Contract vs churn
    st.subheader("Churn by Contract Type")

    contract_churn = pd.crosstab(
        df["Contract"],
        df["Churn"]
    )

    st.bar_chart(contract_churn)

    # Payment method
    st.subheader("Churn by Payment Method")

    payment_churn = pd.crosstab(
        df["PaymentMethod"],
        df["Churn"]
    )

    st.bar_chart(payment_churn)

    # Tenure
    st.subheader("Tenure Distribution")

    fig, ax = plt.subplots()

    sns.histplot(
        data=df,
        x="tenure",
        bins=20,
        kde=True,
        ax=ax
    )

    ax.set_xlabel("Tenure (Months)")
    ax.set_ylabel("Number of Customers")

    st.pyplot(fig)

    # Monthly charges
    st.subheader("Monthly Charges vs Churn")

    fig, ax = plt.subplots()

    sns.boxplot(
        data=df,
        x="Churn",
        y="MonthlyCharges",
        ax=ax
    )

    st.pyplot(fig)

# ---------------------------------------------------
# MACHINE LEARNING MODEL
# ---------------------------------------------------

elif page == "Churn Prediction":

    st.header("🤖 AI / Machine Learning Churn Prediction")

    st.write(
        "The application uses Logistic Regression to predict "
        "whether a customer is likely to churn."
    )

    # Select numerical features
    features = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    X = df[features]
    y = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scaling
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model
    model = LogisticRegression()

    model.fit(
        X_train_scaled,
        y_train
    )

    # Prediction
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    st.subheader("Model Performance")

    st.metric(
        "Model Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    # User input
    st.subheader("Predict Customer Churn")

    col1, col2, col3 = st.columns(3)

    with col1:
        tenure = st.number_input(
            "Tenure (Months)",
            min_value=0,
            max_value=100,
            value=12
        )

    with col2:
        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0
        )

    with col3:
        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=840.0
        )

    if st.button("🔮 Predict Churn"):

        input_data = np.array([
            [
                tenure,
                monthly_charges,
                total_charges
            ]
        ])

        input_scaled = scaler.transform(
            input_data
        )

        prediction = model.predict(
            input_scaled
        )

        probability = model.predict_proba(
            input_scaled
        )[0][1]

        if prediction[0] == 1:

            st.error(
                "⚠️ High Risk: Customer is likely to churn."
            )

        else:

            st.success(
                "✅ Low Risk: Customer is likely to stay."
            )

        st.write(
            f"Estimated churn probability: "
            f"**{probability * 100:.2f}%**"
        )

# ---------------------------------------------------
# AI-POWERED INSIGHTS
# ---------------------------------------------------

elif page == "AI-Powered Insights":

    st.header("💡 AI-Powered Business Insights")

    churn_rate = (
        (df["Churn"] == "Yes").mean() * 100
    )

    avg_monthly_charges = df["MonthlyCharges"].mean()

    avg_tenure = df["tenure"].mean()

    month_to_month_churn = df[
        df["Contract"] == "Month-to-month"
    ]["Churn"].eq("Yes").mean() * 100

    st.subheader("Key Insights")

    st.info(
        f"📌 Overall customer churn rate is "
        f"approximately {churn_rate:.2f}%."
    )

    st.info(
        f"📌 Average monthly customer charges are "
        f"approximately ${avg_monthly_charges:.2f}."
    )

    st.info(
        f"📌 Average customer tenure is "
        f"approximately {avg_tenure:.1f} months."
    )

    st.info(
        f"📌 Month-to-month customers have a churn rate "
        f"of approximately {month_to_month_churn:.2f}%."
    )

    st.subheader("Business Recommendations")

    st.write(
        "1. Identify high-risk customers using the ML model."
    )

    st.write(
        "2. Provide special offers to customers showing "
        "high churn probability."
    )

    st.write(
        "3. Encourage customers to move from month-to-month "
        "contracts to longer-term contracts."
    )

    st.write(
        "4. Monitor customers with short tenure and "
        "higher monthly charges."
    )

    st.success(
        "The combination of Data Analytics and Machine "
        "Learning can help organizations make "
        "data-driven customer retention decisions."
    )

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.write("IBM SkillsBuild Project")
st.sidebar.write("Data Analytics + AI")