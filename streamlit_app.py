import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
DATA_PATH = "data/cleaned_data.xlsx"
RIDGE_MODEL_PATH = "models/ridge_model.pkl"
RIDGE_SCALER_PATH = "models/scaler.pkl"
XGB_MODEL_PATH = "models/xgb_classifier.pkl"


# ---------------------------------------------------------
# Load data and models
# ---------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_excel(DATA_PATH)


@st.cache_resource
def load_models():
    ridge_model = joblib.load(RIDGE_MODEL_PATH)
    ridge_scaler = joblib.load(RIDGE_SCALER_PATH)
    xgb_model = joblib.load(XGB_MODEL_PATH)
    return ridge_model, ridge_scaler, xgb_model


try:
    df = load_data()
    ridge_model, ridge_scaler, xgb_model = load_models()
except Exception as e:
    st.error("Could not load the dataset or trained models.")
    st.code(str(e))
    st.stop()


# ---------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------
def make_features(area, bedrooms, bathrooms, floors, year_built,
                   location_score, school_score, current_price):

    return {
        "Area_sqft": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Floors": floors,
        "Location_Score": location_score,
        "School_Score": school_score,
        "Current_Price": current_price,
        "Price_per_sqft": current_price / area,
        "Area_per_Bedroom": area / bedrooms,
        "Property_age": 2026 - year_built
    }


feature_order = [
    "Area_sqft",
    "Bedrooms",
    "Bathrooms",
    "Floors",
    "Location_Score",
    "School_Score",
    "Current_Price",
    "Price_per_sqft",
    "Area_per_Bedroom",
    "Property_age"
]


# ---------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------
st.sidebar.title(" Real Estate Advisor")
st.sidebar.caption("ML-powered property investment analysis")

page = st.sidebar.radio(
    "Go to",
    [
        " Prediction",
        " Data",
        " Visualizations",
        " Model Insights",
        " About"
    ]
)

st.sidebar.divider()
st.sidebar.caption(
    "Predictions are estimates based on the trained dataset and "
    "should not be treated as financial advice."
)


# =========================================================
# PREDICTION PAGE
# =========================================================
if page == " Prediction":

    st.title(" Prediction")
    st.write(
        "Enter property details to estimate its value after 5 years "
        "and assess its investment potential."
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Property Details")

        area = st.number_input(
            "Area (sq ft)",
            min_value=100.0,
            max_value=10000.0,
            value=2000.0,
            step=50.0
        )

        bedrooms = st.number_input(
            "Bedrooms",
            min_value=1,
            max_value=10,
            value=3,
            step=1
        )

        bathrooms = st.number_input(
            "Bathrooms",
            min_value=1,
            max_value=10,
            value=2,
            step=1
        )

        floors = st.number_input(
            "Floor",
            min_value=0,
            max_value=20,
            value=1,
            step=1
        )

        year_built = st.number_input(
            "Year Built",
            min_value=1900,
            max_value=2026,
            value=2015,
            step=1
        )

    with right:
        st.subheader("Property Scores")

        location_score = st.slider(
            "Location Score",
            0.0, 100.0, 75.0, 1.0
        )

        school_score = st.slider(
            "School Score",
            0.0, 100.0, 70.0, 1.0
        )

        current_price = st.number_input(
            "Current Price",
            min_value=10000.0,
            max_value=10000000.0,
            value=600000.0,
            step=10000.0
        )

    st.divider()

    if st.button(
        "🚀 Analyze Property",
        type="primary",
        use_container_width=True
    ):

        features = make_features(
            area,
            bedrooms,
            bathrooms,
            floors,
            year_built,
            location_score,
            school_score,
            current_price
        )

        X = pd.DataFrame(
            [[features[col] for col in feature_order]],
            columns=feature_order
        )

        # Ridge regression requires scaling.
        X_scaled = ridge_scaler.transform(X)

        predicted_price = float(
            ridge_model.predict(X_scaled)[0]
        )

        # XGBoost was trained on unscaled features.
        investment_prediction = int(
            xgb_model.predict(X)[0]
        )

        # Model confidence/probability.
        probability = xgb_model.predict_proba(X)[0]
        confidence = float(np.max(probability) * 100)

        appreciation = (
            (predicted_price - current_price)
            / current_price
        ) * 100

        # Display headline metrics.
        st.subheader("Investment Verdict")

        m1, m2, m3 = st.columns(3)

        with m1:
            if investment_prediction == 1:
                st.success("✅ Good Investment")
            else:
                st.warning("⚠️ Not Classified as Good Investment")

        with m2:
            st.metric(
                "Model Confidence",
                f"{confidence:.1f}%"
            )

        with m3:
            st.metric(
                "Estimated Price After 5 Years",
                f"₹{predicted_price:,.0f}",
                f"{appreciation:+.2f}%"
            )

        st.divider()

        # Key derived values.
        p1, p2, p3 = st.columns(3)

        with p1:
            st.metric(
                "Price per Sq Ft",
                f"₹{features['Price_per_sqft']:,.2f}"
            )

        with p2:
            st.metric(
                "Area per Bedroom",
                f"{features['Area_per_Bedroom']:,.0f} sq ft"
            )

        with p3:
            st.metric(
                "Property Age",
                f"{features['Property_age']:.0f} years"
            )

        # Five-year trajectory.
        st.subheader("📈 Projected Value Over 5 Years")

        years = np.arange(0, 6)

        # Convert the model's 5-year prediction into a smooth
        # annual trajectory for visualization.
        if predicted_price > 0 and current_price > 0:
            annual_growth = (
                predicted_price / current_price
            ) ** (1 / 5) - 1
        else:
            annual_growth = 0

        projected_values = [
            current_price * ((1 + annual_growth) ** year)
            for year in years
        ]

        chart_df = pd.DataFrame({
            "Year": years,
            "Estimated Value": projected_values
        })

        fig = px.line(
            chart_df,
            x="Year",
            y="Estimated Value",
            markers=True,
            title="Estimated Property Value"
        )

        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Estimated Value",
            hovermode="x unified"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        with st.expander("🔍 View engineered features"):
            st.dataframe(
                pd.DataFrame(
                    [features]
                ).T.rename(columns={0: "Value"}),
                use_container_width=True
            )

        st.info(
            "The classification target was created using a 30% or higher "
            "5-year appreciation threshold during model training. The "
            "classification shown here is produced by the trained XGBoost "
            "model, not by directly applying that threshold to the prediction."
        )


# =========================================================
# DATA PAGE
# =========================================================
elif page == "📊 Data":

    st.title("📊 Dataset Explorer")

    st.write(
        "Explore the cleaned real-estate dataset used for model development."
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Properties", len(df))

    with c2:
        st.metric("Features", len(df.columns))

    with c3:
        st.metric(
            "Average Current Price",
            f"₹{df['Current_Price'].mean():,.0f}"
        )

    with c4:
        st.metric(
            "Average 5-Year Price",
            f"₹{df['Price_After_5_Years'].mean():,.0f}"
        )

    st.divider()

    st.subheader("Dataset Preview")

    rows_to_show = st.slider(
        "Rows to display",
        min_value=5,
        max_value=min(100, len(df)),
        value=10
    )

    st.dataframe(
        df.head(rows_to_show),
        use_container_width=True
    )

    st.subheader("Dataset Information")

    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.write("**Shape**")
        st.write(f"{df.shape[0]} rows × {df.shape[1]} columns")

        st.write("**Missing Values**")
        st.write(
            int(df.isnull().sum().sum())
        )

        st.write("**Duplicate Rows**")
        st.write(
            int(df.duplicated().sum())
        )

    with info_col2:
        st.write("**Data Types**")
        st.dataframe(
            df.dtypes.astype(str).rename("dtype"),
            use_container_width=True
        )

    st.subheader("Summary Statistics")

    st.dataframe(
        df.describe().T,
        use_container_width=True
    )


# =========================================================
# VISUALIZATIONS PAGE
# =========================================================
elif page == "📈 Visualizations":

    st.title("📈 Data Visualizations")
    st.write(
        "Explore relationships between property characteristics and prices."
    )

    tab1, tab2, tab3 = st.tabs([
        "Price Analysis",
        "Property Factors",
        "Correlations"
    ])

    with tab1:

        st.subheader("Current Price by Price Category")

        fig = px.box(
            df,
            x="Price_Category",
            y="Current_Price",
            color="Price_Category",
            title="Current Price Distribution by Category"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Area vs Current Price")

        fig = px.scatter(
            df,
            x="Area_sqft",
            y="Current_Price",
            color="Price_Category",
            hover_data=[
                "Bedrooms",
                "Bathrooms",
                "Location_Score",
                "School_Score"
            ],
            title="Area vs Current Price"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:

        st.subheader("Location Score vs Current Price")

        fig = px.scatter(
            df,
            x="Location_Score",
            y="Current_Price",
            color="Price_Category",
            title="Location Score vs Current Price"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("School Score vs Current Price")

        fig = px.scatter(
            df,
            x="School_Score",
            y="Current_Price",
            color="Price_Category",
            title="School Score vs Current Price"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Bedrooms vs Current Price")

        fig = px.box(
            df,
            x="Bedrooms",
            y="Current_Price",
            color="Bedrooms",
            title="Bedrooms vs Current Price"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:

        st.subheader("Correlation Heatmap")

        numeric_df = df.select_dtypes(
            include=np.number
        ).drop(
            columns=[
                "Property_ID",
                "Price_Year_1",
                "Price_Year_2",
                "Price_Year_3",
                "Price_Year_4"
            ],
            errors="ignore"
        )

        corr = numeric_df.corr()

        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            title="Feature Correlation Matrix"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.info(
            "Correlation measures association, not causation. A high or low "
            "correlation does not by itself prove that one variable causes "
            "changes in another."
        )


# =========================================================
# MODEL INSIGHTS PAGE
# =========================================================
elif page == "🤖 Model Insights":

    st.title("🤖 Model Insights")

    st.write(
        "Summary of the final machine-learning models used in the application."
    )

    reg_col, cls_col = st.columns(2)

    with reg_col:
        st.subheader("📌 Regression Model")

        st.markdown("""
        **Model:** Ridge Regression

        **Target:** `Price_After_5_Years`

        **Purpose:** Estimate the property's price after five years.
        """)

        metrics = pd.DataFrame({
            "Metric": ["MAE", "RMSE", "R²"],
            "Value": [72137.42, 86560.49, 0.86]
        })

        st.dataframe(
            metrics,
            hide_index=True,
            use_container_width=True
        )

        st.caption(
            "Lower MAE/RMSE are better. Higher R² is better."
        )

    with cls_col:
        st.subheader("📌 Classification Model")

        st.markdown("""
        **Model:** XGBoost Classifier

        **Target:** `Good_Investment`

        **Purpose:** Classify whether a property is a good investment.
        """)

        metrics = pd.DataFrame({
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "F1 Macro",
                "Balanced Accuracy"
            ],
            "Value": [
                0.6300,
                0.6795,
                0.8154,
                0.7413,
                0.5461,
                0.5505
            ]
        })

        st.dataframe(
            metrics,
            hide_index=True,
            use_container_width=True
        )

    st.divider()

    st.subheader("Regression Features")

    st.write(
        ", ".join(feature_order)
    )

    st.subheader("Classification Target")

    st.write(
        "A property was labelled Good Investment when its observed "
        "5-year appreciation was at least 30%."
    )

    st.warning(
        "The classifier has a relatively low balanced accuracy and F1 Macro "
        "because the model is much better at identifying the positive class "
        "than the negative class. Therefore, this model should be presented "
        "as an academic/project prediction rather than a financial decision tool."
    )

    st.subheader("Classification Confusion Matrix")

    cm = np.array([
        [10, 25],
        [12, 53]
    ])

    cm_df = pd.DataFrame(
        cm,
        index=["Actual Not Good", "Actual Good"],
        columns=["Predicted Not Good", "Predicted Good"]
    )

    fig = px.imshow(
        cm_df,
        text_auto=True,
        aspect="auto",
        title="XGBoost Confusion Matrix"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# ABOUT PAGE
# =========================================================
else:

    st.title("ℹ️ About the Project")

    st.markdown("""
    ## Real Estate Investment Advisor

    This project combines **machine learning, feature engineering,
    exploratory data analysis and Streamlit** to analyze real-estate
    investment potential.

    ### Main objectives

    - Predict the estimated property price after 5 years.
    - Classify whether a property is a good investment.
    - Explore factors associated with property prices.
    - Provide an interactive interface for predictions.

    ### Technologies

    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - XGBoost
    - Joblib
    - Streamlit
    - Plotly
    - MLflow

    ### Feature engineering

    The project creates:

    - Price per square foot
    - Area per bedroom
    - Property age

    ### Model pipeline

    **Ridge Regression**

    Property details → Feature engineering → StandardScaler →
    Ridge Regression → Estimated 5-year price

    **XGBoost Classification**

    Property details → Feature engineering → XGBoost →
    Good Investment classification

    ### Important note

    This application is an educational machine-learning project.
    Its predictions are estimates based on the supplied dataset and
    should not be interpreted as professional financial advice.
    """)

    st.success("Project ready for demonstration and deployment.")
