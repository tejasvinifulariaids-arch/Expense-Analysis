import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Expense Analysis Project",
    page_icon="💰",
    layout="wide"
)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv("expenses.csv")


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("💰 Expense Project")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📁 Data File",
        "🧹 Data Cleaning",
        "📊 Data Visualization",
        "🤖 Model Creation",
        "🔮 Predict"
    ]
)


# =========================================================
# HOME PAGE
# =========================================================

if page == "🏠 Home":

    st.title("💰 Personal Expense Analysis")

    st.write(
        "This project analyzes my personal expense data "
        "using Python, Pandas, Streamlit and Machine Learning."
    )

    st.markdown("---")

    st.header("📌 Project Overview")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("What this project does")

        st.write("""
        - Loads the expense dataset
        - Checks and cleans the data
        - Performs data analysis
        - Creates visualizations
        - Builds a machine learning model
        - Predicts a possible expense amount
        """)

    with col2:

        st.subheader("Tools Used")

        st.write("""
        🐍 Python

        🐼 Pandas

        📊 Matplotlib

        🎈 Streamlit

        🤖 Scikit-learn
        """)

    st.markdown("---")

    st.header("📊 Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric("Total Spending", f"₹{df['INR'].sum():,.0f}")


# =========================================================
# DATA FILE PAGE
# =========================================================

elif page == "📁 Data File":

    st.title("📁 Data File")

    st.write(
        "This page displays the original dataset "
        "that I downloaded from Kaggle."
    )

    st.subheader("Dataset")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Dataset Shape")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Number of rows:")
        st.write(df.shape[0])

    with col2:
        st.write("Number of columns:")
        st.write(df.shape[1])

    st.subheader("Column Names")

    st.write(list(df.columns))


# =========================================================
# DATA CLEANING PAGE
# =========================================================

elif page == "🧹 Data Cleaning":

    st.title("🧹 Data Cleaning")

    st.write(
        "Before analysis, I check the dataset for "
        "missing values and duplicate rows."
    )

    # Missing values
    st.subheader("1️⃣ Missing Values")

    missing_values = df.isnull().sum()

    missing_data = pd.DataFrame({
        "Column": missing_values.index,
        "Missing Values": missing_values.values
    })

    st.dataframe(
        missing_data,
        use_container_width=True,
        hide_index=True
    )

    # Duplicate values
    st.subheader("2️⃣ Duplicate Rows")

    duplicate_count = df.duplicated().sum()

    st.write(
        f"Number of duplicate rows: **{duplicate_count}**"
    )

    # Clean data
    st.subheader("3️⃣ Clean the Data")

    clean_df = df.copy()

    # Convert Date column
    clean_df["Date"] = pd.to_datetime(
        clean_df["Date"],
        errors="coerce"
    )

    # Convert INR to number
    clean_df["INR"] = pd.to_numeric(
        clean_df["INR"],
        errors="coerce"
    )

    # Remove rows where INR is missing
    clean_df = clean_df.dropna(
        subset=["INR"]
    )

    # Remove duplicate rows
    clean_df = clean_df.drop_duplicates()

    st.write(
        f"Original rows: **{len(df)}**"
    )

    st.write(
        f"Rows after cleaning: **{len(clean_df)}**"
    )

    st.subheader("Cleaned Data")

    st.dataframe(
        clean_df,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# DATA VISUALIZATION PAGE
# =========================================================

elif page == "📊 Data Visualization":

    st.title("📊 Data Visualization")

    st.write(
        "Here I use charts to understand my spending habits."
    )

    # Prepare data
    chart_df = df.copy()

    chart_df["INR"] = pd.to_numeric(
        chart_df["INR"],
        errors="coerce"
    )

    chart_df = chart_df.dropna(
        subset=["INR"]
    )

    # -----------------------------------------
    # Spending by Category
    # -----------------------------------------

    st.subheader("💸 Spending by Category")

    category_data = (
        chart_df
        .groupby("Category")["INR"]
        .sum()
        .sort_values(ascending=False)
    )

    st.bar_chart(category_data)

    # -----------------------------------------
    # Spending by Account
    # -----------------------------------------

    st.subheader("💳 Spending by Account")

    account_data = (
        chart_df
        .groupby("Account")["INR"]
        .sum()
        .sort_values(ascending=False)
    )

    st.bar_chart(account_data)

    # -----------------------------------------
    # Top Categories
    # -----------------------------------------

    st.subheader("🏆 Top Spending Categories")

    top_categories = category_data.head(5)

    st.bar_chart(top_categories)

    # -----------------------------------------
    # Monthly Spending
    # -----------------------------------------

    st.subheader("📅 Monthly Spending")

    chart_df["Date"] = pd.to_datetime(
        chart_df["Date"],
        errors="coerce"
    )

    chart_df["Month"] = (
        chart_df["Date"]
        .dt.to_period("M")
        .astype(str)
    )

    monthly_data = (
        chart_df
        .groupby("Month")["INR"]
        .sum()
    )

    st.line_chart(monthly_data)


# =========================================================
# MODEL CREATION PAGE
# =========================================================

elif page == "🤖 Model Creation":

    st.title("🤖 Model Creation")

    st.write(
        "Here I create a simple machine learning model "
        "to predict an expense amount."
    )

    st.subheader("🎯 Prediction Target")

    st.write(
        "The model will try to predict the **INR amount** "
        "based on information such as category, account, "
        "month and day."
    )

    # -----------------------------------------
    # Prepare data
    # -----------------------------------------

    model_df = df.copy()

    model_df["Date"] = pd.to_datetime(
        model_df["Date"],
        errors="coerce"
    )

    model_df["INR"] = pd.to_numeric(
        model_df["INR"],
        errors="coerce"
    )

    # Remove missing values
    model_df = model_df.dropna(
        subset=["Date", "INR", "Category", "Account"]
    )

    # Create month and day columns
    model_df["Month"] = model_df["Date"].dt.month
    model_df["Day"] = model_df["Date"].dt.day

    # Select columns
    model_df = model_df[
        [
            "Category",
            "Account",
            "Month",
            "Day",
            "INR"
        ]
    ]

    # Convert text columns to numbers
    model_df = pd.get_dummies(
        model_df,
        columns=["Category", "Account"]
    )

    # X = input data
    X = model_df.drop("INR", axis=1)

    # y = value we want to predict
    y = model_df["INR"]

    # -----------------------------------------
    # Train and test data
    # -----------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # -----------------------------------------
    # Create model
    # -----------------------------------------

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    # -----------------------------------------
    # Make predictions
    # -----------------------------------------

    predictions = model.predict(X_test)

    # -----------------------------------------
    # Model performance
    # -----------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    st.subheader("📈 Model Performance")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Mean Absolute Error",
            f"₹{mae:,.2f}"
        )

    with col2:

        st.metric(
            "R² Score",
            f"{r2:.2f}"
        )

    st.subheader("🔍 Actual vs Predicted")

    result = pd.DataFrame({
        "Actual Amount": y_test.values,
        "Predicted Amount": predictions
    })

    st.dataframe(
        result.head(20),
        use_container_width=True,
        hide_index=True
    )

    st.success(
        "Model created successfully!"
    )


# =========================================================
# PREDICTION PAGE
# =========================================================

elif page == "🔮 Predict":

    st.title("🔮 Predict Expense")

    st.write(
        "Enter some information below and the machine "
        "learning model will predict the expense amount."
    )

    # -----------------------------------------
    # Prepare model data
    # -----------------------------------------

    model_df = df.copy()

    model_df["Date"] = pd.to_datetime(
        model_df["Date"],
        errors="coerce"
    )

    model_df["INR"] = pd.to_numeric(
        model_df["INR"],
        errors="coerce"
    )

    model_df = model_df.dropna(
        subset=["Date", "INR", "Category", "Account"]
    )

    model_df["Month"] = model_df["Date"].dt.month
    model_df["Day"] = model_df["Date"].dt.day

    model_df = model_df[
        [
            "Category",
            "Account",
            "Month",
            "Day",
            "INR"
        ]
    ]

    # -----------------------------------------
    # Convert categories into numbers
    # -----------------------------------------

    model_df = pd.get_dummies(
        model_df,
        columns=["Category", "Account"]
    )

    X = model_df.drop("INR", axis=1)
    y = model_df["INR"]

    # Train model
    model = LinearRegression()

    model.fit(X, y)

    # -----------------------------------------
    # User Input
    # -----------------------------------------

    st.subheader("Enter Expense Details")

    col1, col2 = st.columns(2)

    with col1:

        category = st.selectbox(
            "Category",
            df["Category"]
            .dropna()
            .unique()
        )

    with col2:

        account = st.selectbox(
            "Account",
            df["Account"]
            .dropna()
            .unique()
        )

    col3, col4 = st.columns(2)

    with col3:

        month = st.number_input(
            "Month",
            min_value=1,
            max_value=12,
            value=1
        )

    with col4:

        day = st.number_input(
            "Day",
            min_value=1,
            max_value=31,
            value=1
        )

    # -----------------------------------------
    # Create input data
    # -----------------------------------------

    input_data = pd.DataFrame({
        "Month": [month],
        "Day": [day]
    })

    # Add category columns
    for column in X.columns:

        if column.startswith("Category_"):

            input_data[column] = 0

    # Add account columns
    for column in X.columns:

        if column.startswith("Account_"):

            input_data[column] = 0

    # Set selected category to 1
    category_column = "Category_" + category

    if category_column in input_data.columns:

        input_data[category_column] = 1

    # Set selected account to 1
    account_column = "Account_" + account

    if account_column in input_data.columns:

        input_data[account_column] = 1

    # Make sure columns are in same order
    input_data = input_data.reindex(
        columns=X.columns,
        fill_value=0
    )

    # -----------------------------------------
    # Prediction Button
    # -----------------------------------------

    if st.button(
        "🔮 Predict Expense",
        type="primary"
    ):

        prediction = model.predict(
            input_data
        )

        predicted_amount = prediction[0]

        if predicted_amount < 0:
            predicted_amount = 0

        st.success(
            f"Predicted Expense: ₹{predicted_amount:,.2f}"
        )

        st.info(
            "This is a machine learning prediction "
            "and may not be exactly equal to the actual expense."
        )
