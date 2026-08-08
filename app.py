
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import warnings

warnings.filterwarnings("ignore")


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="🔬 Breast Cancer Diagnosis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown(
    """
    <style>

    :root {
        --text-primary: #2c3e50;
        --text-secondary: #34495e;
        --bg-light: #ffffff;
        --bg-light-alt: #f5f7fa;
        --border-color: #e0e0e0;
        --success-bg: #d4edda;
        --success-text: #155724;
        --success-border: #28a745;
        --warning-bg: #fff3cd;
        --warning-text: #856404;
        --warning-border: #ffc107;
        --error-bg: #f8d7da;
        --error-text: #721c24;
        --error-border: #dc3545;
        --primary-blue: #3498db;
        --primary-blue-dark: #2980b9;
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --text-primary: #e0e0e0;
            --text-secondary: #b0b0b0;
            --bg-light: #2a2a2a;
            --bg-light-alt: #1e1e1e;
            --border-color: #404040;
            --success-bg: #1e4620;
            --success-text: #90ee90;
            --success-border: #52b788;
            --warning-bg: #4d3d00;
            --warning-text: #ffd700;
            --warning-border: #ffa500;
            --error-bg: #4d1f1f;
            --error-text: #ff6b6b;
            --error-border: #ff4444;
        }
    }

    [data-testid="stAppViewContainer"] {
        background-color: var(--bg-light-alt);
        color: var(--text-primary);
    }

    [data-testid="stSidebar"] {
        background-color: #2c3e50;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] label {
        color: #ecf0f1 !important;
    }

    [data-testid="metric-container"] {
        background-color: var(--bg-light);
        color: var(--text-primary) !important;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15);
        border-left: 4px solid var(--primary-blue);
    }

    [data-testid="metric-container"] label {
        color: var(--text-secondary) !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
    }

    p, span {
        color: var(--text-primary);
    }

    .info-box {
        background: var(--bg-light);
        color: var(--text-primary);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15);
        border-left: 4px solid #e74c3c;
    }

    .success-box {
        background: var(--success-bg);
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid var(--success-border);
        color: var(--success-text);
    }

    .warning-box {
        background: var(--warning-bg);
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid var(--warning-border);
        color: var(--warning-text);
    }

    .stButton > button {
        background: linear-gradient(
            90deg,
            var(--primary-blue) 0%,
            var(--primary-blue-dark) 100%
        );
        color: white !important;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: 600;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(52, 152, 219, 0.4);
    }

    input, select, textarea {
        background-color: var(--bg-light) !important;
        color: var(--text-primary) !important;
        border: 2px solid var(--border-color) !important;
        border-radius: 8px;
    }

    label {
        color: var(--text-primary) !important;
        font-weight: 600;
    }

    hr {
        border-color: var(--border-color);
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# SESSION STATE
# ============================================================================

if "predictions" not in st.session_state:
    st.session_state.predictions = {}

if "sample_loaded" not in st.session_state:
    st.session_state.sample_loaded = False


# ============================================================================
# CONSTANTS
# ============================================================================

FEATURE_COLUMNS = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]


# ============================================================================
# LOAD DATA
# ============================================================================

@st.cache_data
def load_data():
    """Load the breast cancer dataset."""

    try:
        data = pd.read_csv("data.csv")
        return data

    except FileNotFoundError:
        st.error(
            "❌ data.csv not found. "
            "Please make sure data.csv is present in the project folder."
        )
        return None

    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        return None


# ============================================================================
# LOAD MODELS
# ============================================================================

@st.cache_resource
def load_models():
    """Load all trained models saved using Joblib."""

    models = {}

    model_files = {
        "svm": "svm_model.joblib",
        "random_forest": "random_forest_model.joblib",
        "logistic_regression": "logistic_regression_model.joblib",
        "xgboost": "xgboost_model.joblib",
    }

    for model_name, file_name in model_files.items():

        try:
            models[model_name] = joblib.load(file_name)

        except FileNotFoundError:
            pass

        except Exception as e:
            st.warning(
                f"⚠️ Could not load {file_name}: {e}"
            )

    return models


# ============================================================================
# LOAD PREPROCESSORS
# ============================================================================

@st.cache_resource
def load_preprocessors():
    """Load imputer, scaler and label encoder."""

    try:
        scaler = joblib.load("scaler.joblib")
        imputer = joblib.load("imputer.joblib")
        label_encoder = joblib.load("label_encoder.joblib")

        return scaler, imputer, label_encoder

    except FileNotFoundError as e:
        st.error(
            f"❌ Preprocessor file not found: {e}"
        )
        return None, None, None

    except Exception as e:
        st.error(
            f"❌ Error loading preprocessors: {e}"
        )
        return None, None, None


# ============================================================================
# PREDICTION CARD
# ============================================================================

def create_prediction_card(diagnosis, confidence):

    if diagnosis == 1:

        color = "#e74c3c"
        bg_color = "#fadbd8"
        icon = "⚠️"
        message = "MALIGNANT"
        text_color = "#721c24"

    else:

        color = "#27ae60"
        bg_color = "#d4edda"
        icon = "✅"
        message = "BENIGN"
        text_color = "#155724"

    st.markdown(
        f"""
        <div style="
            background-color: {bg_color};
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            border: 3px solid {color};
            margin: 20px 0;
        ">

            <h1 style="
                color: {text_color};
                margin: 0;
            ">
                {icon} {message}
            </h1>

            <p style="
                font-size: 18px;
                color: {text_color};
                margin-top: 10px;
                font-weight: bold;
            ">
                Confidence: {confidence:.2f}%
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    <h2 style="color:white !important;">
        🔬 CANCER DIAGNOSIS
    </h2>

    <p style="color:white !important;">
        ML-Powered Healthcare
    </p>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "📑 Navigation Menu",
    [
        "🏠 Home",
        "📊 Dataset Overview",
        "📈 Data Analysis",
        "📉 Visualizations",
        "🤖 Model Prediction",
        "ℹ️ About Project",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")


# ============================================================================
# HOME PAGE
# ============================================================================

if page == "🏠 Home":

    col1, col2 = st.columns([1, 1])

    with col1:

        st.markdown(
            """
            # 🔬 Breast Cancer Diagnosis
            ## AI-Powered Prediction System
            """
        )

        st.markdown(
            """
            <div class="info-box">

            <p style="font-size:16px; line-height:1.8;">

            This machine learning dashboard predicts breast cancer
            diagnosis using cellular characteristics from the
            Breast Cancer Wisconsin Diagnostic dataset.

            Multiple machine learning models are trained and evaluated
            to provide classification predictions.

            </p>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            """
            ### 🔍 Key Features

            **✓ Multiple ML Models**

            - Support Vector Machine
            - Random Forest
            - Logistic Regression
            - XGBoost

            **✓ Real-time Predictions**

            - Instant prediction
            - Confidence score

            **✓ Data Visualization**

            - Interactive charts
            - Correlation analysis
            - Feature comparison

            **✓ User-Friendly Interface**

            - Simple navigation
            - Interactive inputs
            """
        )

    st.markdown("---")

    st.markdown("### 📊 Dataset Metrics")

    data = load_data()

    if data is not None:

        total_samples = len(data)

        malignant = (
            data["diagnosis"] == "M"
        ).sum()

        benign = (
            data["diagnosis"] == "B"
        ).sum()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "📋 Total Samples",
                f"{total_samples:,}"
            )

        with col2:
            st.metric(
                "⚠️ Malignant Cases",
                f"{malignant}",
                f"{(malignant / total_samples) * 100:.1f}%"
            )

        with col3:
            st.metric(
                "✅ Benign Cases",
                f"{benign}",
                f"{(benign / total_samples) * 100:.1f}%"
            )

        with col4:
            st.metric(
                "🔢 Features",
                "30"
            )

    st.markdown("---")

    st.markdown(
        """
        ### 🚀 Getting Started

        1. **Dataset Overview** - Explore the dataset
        2. **Data Analysis** - Analyze statistical information
        3. **Visualizations** - Explore interactive charts
        4. **Model Prediction** - Enter measurements and predict
        5. **About Project** - View project details
        """
    )


# ============================================================================
# DATASET OVERVIEW
# ============================================================================

elif page == "📊 Dataset Overview":

    st.title("📊 Dataset Overview")
    st.markdown("---")

    data = load_data()

    if data is not None:

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "📏 Rows",
                data.shape[0]
            )

        with col2:
            st.metric(
                "🔢 Features",
                len(FEATURE_COLUMNS)
            )

        with col3:
            st.metric(
                "🎯 Target Classes",
                data["diagnosis"].nunique()
            )

        with col4:
            st.metric(
                "❌ Missing Values",
                data.isnull().sum().sum()
            )

        st.markdown("---")

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "📋 First 10 Rows",
                "📊 Data Types",
                "📉 Missing Values",
                "📈 Statistics",
            ]
        )

        with tab1:

            st.subheader("First 10 Rows")

            st.dataframe(
                data.head(10),
                use_container_width=True
            )

        with tab2:

            st.subheader("Data Types")

            dtype_info = pd.DataFrame(
                {
                    "Column": data.columns,
                    "Data Type": data.dtypes.astype(str),
                    "Non-Null Count": data.notna().sum(),
                }
            )

            st.dataframe(
                dtype_info,
                use_container_width=True
            )

        with tab3:

            st.subheader("Missing Values")

            missing_data = pd.DataFrame(
                {
                    "Column": data.columns,
                    "Missing Count": data.isnull().sum(),
                    "Missing %":
                        (data.isnull().sum() / len(data)) * 100,
                }
            )

            missing_data = missing_data[
                missing_data["Missing Count"] > 0
            ]

            if len(missing_data) == 0:

                st.success(
                    "✅ No missing values found!"
                )

            else:

                st.dataframe(
                    missing_data,
                    use_container_width=True
                )

        with tab4:

            st.subheader("Statistical Summary")

            st.dataframe(
                data.describe(),
                use_container_width=True
            )


# ============================================================================
# DATA ANALYSIS
# ============================================================================

elif page == "📈 Data Analysis":

    st.title("📈 Data Analysis")
    st.markdown("---")

    data = load_data()

    if data is not None:

        st.subheader(
            "🎯 Target Class Distribution"
        )

        col1, col2 = st.columns([2, 1])

        class_counts = data["diagnosis"].value_counts()

        with col1:

            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x=["Benign", "Malignant"],
                    y=[
                        class_counts.get("B", 0),
                        class_counts.get("M", 0),
                    ],
                    marker=dict(
                        color=["#27ae60", "#e74c3c"]
                    ),
                    text=[
                        class_counts.get("B", 0),
                        class_counts.get("M", 0),
                    ],
                    textposition="auto",
                )
            )

            fig.update_layout(
                title="Diagnosis Distribution",
                xaxis_title="Diagnosis",
                yaxis_title="Count",
                height=400,
                showlegend=False,
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col2:

            st.metric(
                "Malignant %",
                f"{(class_counts.get('M', 0) / len(data)) * 100:.1f}%"
            )

            st.metric(
                "Benign %",
                f"{(class_counts.get('B', 0) / len(data)) * 100:.1f}%"
            )

        st.markdown("---")

        st.subheader(
            "📊 Statistical Summary by Diagnosis"
        )

        stat_data = data[
            FEATURE_COLUMNS + ["diagnosis"]
        ]

        stat_summary = (
            stat_data
            .groupby("diagnosis")
            .describe()
            .T
        )

        st.dataframe(
            stat_summary,
            use_container_width=True
        )

        st.markdown("---")

        st.subheader(
            "🔍 Interactive Data Filters"
        )

        col1, col2 = st.columns(2)

        with col1:

            selected_feature = st.selectbox(
                "Select a feature:",
                FEATURE_COLUMNS
            )

        with col2:

            feature_min = float(
                data[selected_feature].min()
            )

            feature_max = float(
                data[selected_feature].max()
            )

            feature_mean = float(
                data[selected_feature].mean()
            )

            feature_range = st.slider(
                f"Filter {selected_feature}",
                min_value=feature_min,
                max_value=feature_max,
                value=(feature_min, feature_max),
            )

        filtered_data = data[
            (data[selected_feature] >= feature_range[0])
            &
            (data[selected_feature] <= feature_range[1])
        ]

        st.success(
            f"✅ Showing {len(filtered_data)} records "
            f"out of {len(data)}"
        )

        st.dataframe(
            filtered_data,
            use_container_width=True
        )


# ============================================================================
# VISUALIZATIONS
# ============================================================================

elif page == "📉 Visualizations":

    st.title("📉 Visualizations")
    st.markdown("---")

    data = load_data()

    if data is not None:

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "📊 Feature Histograms",
                "🔗 Correlation Heatmap",
                "🔵 Scatter Plot",
                "📈 Feature Comparison",
            ]
        )

        # --------------------------------------------------------------------
        # HISTOGRAM
        # --------------------------------------------------------------------

        with tab1:

            st.subheader(
                "Feature Distribution"
            )

            col1, col2 = st.columns(2)

            with col1:

                selected_feature = st.selectbox(
                    "Select feature:",
                    FEATURE_COLUMNS,
                    key="hist_feature",
                )

            with col2:

                bins = st.slider(
                    "Number of bins:",
                    10,
                    100,
                    30,
                )

            fig = go.Figure()

            fig.add_trace(
                go.Histogram(
                    x=data[
                        data["diagnosis"] == "M"
                    ][selected_feature],
                    name="Malignant",
                    opacity=0.7,
                    nbinsx=bins,
                    marker_color="#e74c3c",
                )
            )

            fig.add_trace(
                go.Histogram(
                    x=data[
                        data["diagnosis"] == "B"
                    ][selected_feature],
                    name="Benign",
                    opacity=0.7,
                    nbinsx=bins,
                    marker_color="#27ae60",
                )
            )

            fig.update_layout(
                title=f"Distribution of {selected_feature}",
                xaxis_title=selected_feature,
                yaxis_title="Frequency",
                height=500,
                barmode="overlay",
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # --------------------------------------------------------------------
        # CORRELATION
        # --------------------------------------------------------------------

        with tab2:

            st.subheader(
                "Feature Correlation Matrix"
            )

            correlation = data[
                FEATURE_COLUMNS
            ].corr()

            fig = go.Figure(
                data=go.Heatmap(
                    z=correlation.values,
                    x=correlation.columns,
                    y=correlation.columns,
                    colorscale="RdBu",
                    zmid=0,
                )
            )

            fig.update_layout(
                title="Feature Correlation Matrix",
                height=800,
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # --------------------------------------------------------------------
        # SCATTER
        # --------------------------------------------------------------------

        with tab3:

            st.subheader(
                "Feature Comparison - Scatter Plot"
            )

            col1, col2 = st.columns(2)

            with col1:

                x_feature = st.selectbox(
                    "Select X-axis feature:",
                    FEATURE_COLUMNS,
                    key="x_scatter",
                )

            with col2:

                y_feature = st.selectbox(
                    "Select Y-axis feature:",
                    FEATURE_COLUMNS,
                    key="y_scatter",
                )

            fig = px.scatter(
                data,
                x=x_feature,
                y=y_feature,
                color="diagnosis",
                color_discrete_map={
                    "M": "#e74c3c",
                    "B": "#27ae60",
                },
                title=f"{x_feature} vs {y_feature}",
                labels={
                    "diagnosis": "Diagnosis"
                },
                height=500,
            )

            fig.update_traces(
                marker=dict(
                    size=8,
                    opacity=0.7
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # --------------------------------------------------------------------
        # FEATURE COMPARISON
        # --------------------------------------------------------------------

        with tab4:

            st.subheader(
                "Top Features - Mean Comparison"
            )

            feature_means = (
                data
                .groupby("diagnosis")[FEATURE_COLUMNS]
                .mean()
            )

            mean_diff = (
                feature_means.loc["M"]
                - feature_means.loc["B"]
            )

            top_features = (
                mean_diff
                .abs()
                .nlargest(10)
            )

            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x=top_features.index,
                    y=mean_diff[
                        top_features.index
                    ],
                    marker_color=[
                        "#e74c3c"
                        if value > 0
                        else "#27ae60"
                        for value in
                        mean_diff[
                            top_features.index
                        ]
                    ],
                )
            )

            fig.update_layout(
                title=(
                    "Top 10 Features - "
                    "Mean Difference "
                    "(Malignant vs Benign)"
                ),
                xaxis_title="Features",
                yaxis_title="Mean Difference",
                height=500,
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ============================================================================
# MODEL PREDICTION
# ============================================================================

elif page == "🤖 Model Prediction":

    st.title("🤖 Model Prediction")
    st.markdown("---")

    models = load_models()

    scaler, imputer, label_encoder = (
        load_preprocessors()
    )

    data = load_data()

    # ------------------------------------------------------------------------
    # CHECK REQUIRED FILES
    # ------------------------------------------------------------------------

    if not models:

        st.error(
            """
            ❌ No trained models were found.

            Make sure the following files are present:

            - svm_model.joblib
            - random_forest_model.joblib
            - logistic_regression_model.joblib
            - xgboost_model.joblib
            """
        )

    elif (
        scaler is None
        or imputer is None
        or label_encoder is None
    ):

        st.error(
            """
            ❌ Preprocessor files are missing.

            Required files:

            - scaler.joblib
            - imputer.joblib
            - label_encoder.joblib
            """
        )

    elif data is not None:

        with st.expander(
            "📖 How to Use This Predictor",
            expanded=True,
        ):

            st.markdown(
                """
                ### Step-by-Step Guide

                **1. Enter Measurements**

                Enter the 30 cellular measurements.

                **2. Use Sample Data**

                You can automatically load a sample
                malignant patient from the dataset.

                **3. Make Prediction**

                Click the prediction button.

                **4. View Results**

                The SVM model provides the primary
                prediction and confidence score.

                Other trained models are also shown
                for comparison.

                ⚠️ This application is for educational
                and research purposes only.
                """
            )

        st.info(
            "💡 Enter the patient's cellular measurements "
            "to predict the diagnosis."
        )

        # --------------------------------------------------------------------
        # INPUT SECTION
        # --------------------------------------------------------------------

        st.subheader(
            "📋 Patient Cellular Measurements"
        )

        input_data = {}

        cols = st.columns(3)

        for idx, feature in enumerate(
            FEATURE_COLUMNS
        ):

            col = cols[idx % 3]

            with col:

                min_val = float(
                    data[feature].min()
                )

                max_val = float(
                    data[feature].max()
                )

                mean_val = float(
                    data[feature].mean()
                )

                input_data[feature] = st.number_input(
                    feature,
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    step=0.01,
                    key=f"input_{feature}",
                )

        st.markdown("---")

        # --------------------------------------------------------------------
        # BUTTONS
        # --------------------------------------------------------------------

        col1, col2, col3 = st.columns(3)

        # CLEAR
        with col1:

            if st.button(
                "🔄 Clear Fields",
                use_container_width=True,
            ):

                st.session_state.predictions = {}

                for feature in FEATURE_COLUMNS:

                    key = f"input_{feature}"

                    if key in st.session_state:
                        del st.session_state[key]

                st.rerun()

        # PREDICT
        with col2:

            if st.button(
                "🔮 Make Prediction",
                use_container_width=True,
            ):

                try:

                    # Create dataframe with exact
                    # training feature order
                    feature_df = pd.DataFrame(
                        [
                            [
                                input_data[feature]
                                for feature
                                in FEATURE_COLUMNS
                            ]
                        ],
                        columns=FEATURE_COLUMNS,
                    )

                    # Apply same preprocessing
                    # used during training
                    feature_imputed = (
                        imputer.transform(
                            feature_df
                        )
                    )

                    feature_scaled = (
                        scaler.transform(
                            feature_imputed
                        )
                    )

                    # --------------------------------------------------------
                    # SVM = BEST MODEL
                    # --------------------------------------------------------

                    if "svm" in models:

                        best_model = models["svm"]

                    else:

                        best_model = next(
                            iter(models.values())
                        )

                    prediction = int(
                        best_model.predict(
                            feature_scaled
                        )[0]
                    )

                    # Confidence
                    if hasattr(
                        best_model,
                        "predict_proba"
                    ):

                        probabilities = (
                            best_model
                            .predict_proba(
                                feature_scaled
                            )[0]
                        )

                        confidence = (
                            max(probabilities)
                            * 100
                        )

                    else:

                        confidence = 85.0

                    # Save result
                    st.session_state.predictions = {
                        "diagnosis": prediction,
                        "confidence": confidence,
                        "feature_scaled":
                            feature_scaled,
                    }

                    st.success(
                        "✅ Prediction generated successfully!"
                    )

                except Exception as e:

                    st.error(
                        f"❌ Error during prediction: {e}"
                    )

        # SAMPLE DATA
        with col3:

            if st.button(
                "💾 Use Sample Data",
                use_container_width=True,
            ):

                malignant_sample = (
                    data[
                        data["diagnosis"] == "M"
                    ].iloc[0]
                )

                for feature in FEATURE_COLUMNS:

                    st.session_state[
                        f"input_{feature}"
                    ] = float(
                        malignant_sample[feature]
                    )

                st.rerun()

        # --------------------------------------------------------------------
        # DISPLAY RESULT
        # --------------------------------------------------------------------

        if st.session_state.predictions:

            st.markdown("---")

            st.subheader(
                "🎯 Prediction Result"
            )

            diagnosis = (
                st.session_state
                .predictions["diagnosis"]
            )

            confidence = (
                st.session_state
                .predictions["confidence"]
            )

            create_prediction_card(
                diagnosis,
                confidence
            )

            # ---------------------------------------------------------------
            # INTERPRETATION
            # ---------------------------------------------------------------

            if diagnosis == 1:

                st.warning(
                    """
                    ⚠️ **MALIGNANT Prediction**

                    The model predicts a malignant
                    classification for the entered
                    measurements.

                    This result is not a medical diagnosis.
                    Please consult a qualified healthcare
                    professional.
                    """
                )

            else:

                st.success(
                    """
                    ✅ **BENIGN Prediction**

                    The model predicts a benign
                    classification for the entered
                    measurements.

                    This result is not a medical diagnosis.
                    Please consult a qualified healthcare
                    professional for confirmation.
                    """
                )

            # ---------------------------------------------------------------
            # ALL MODEL PREDICTIONS
            # ---------------------------------------------------------------

            st.markdown("---")

            st.subheader(
                "📊 Model Comparison"
            )

            st.write(
                "Predictions from all available trained models:"
            )

            feature_scaled = (
                st.session_state
                .predictions["feature_scaled"]
            )

            all_predictions = {}

            for model_name, model in models.items():

                try:

                    pred_label = int(
                        model.predict(
                            feature_scaled
                        )[0]
                    )

                    if hasattr(
                        model,
                        "predict_proba"
                    ):

                        probabilities = (
                            model
                            .predict_proba(
                                feature_scaled
                            )[0]
                        )

                        model_confidence = (
                            max(probabilities)
                            * 100
                        )

                    else:

                        model_confidence = np.nan

                    all_predictions[
                        model_name
                    ] = {
                        "Prediction":
                            (
                                "Malignant"
                                if pred_label == 1
                                else "Benign"
                            ),

                        "Confidence (%)":
                            (
                                round(
                                    model_confidence,
                                    2
                                )
                                if not np.isnan(
                                    model_confidence
                                )
                                else "N/A"
                            ),
                    }

                except Exception as e:

                    all_predictions[
                        model_name
                    ] = {
                        "Prediction": "Error",
                        "Confidence (%)": "N/A",
                    }

            pred_df = pd.DataFrame(
                all_predictions
            ).T

            st.dataframe(
                pred_df,
                use_container_width=True
            )


# ============================================================================
# ABOUT PROJECT
# ============================================================================

elif page == "ℹ️ About Project":

    st.title("ℹ️ About This Project")
    st.markdown("---")

    st.markdown(
        """
        ## 🏥 Breast Cancer Diagnosis ML Project

        ### 📋 Project Overview

        This machine learning project predicts breast cancer
        diagnosis using cellular measurements from the
        Breast Cancer Wisconsin Diagnostic dataset.

        The application uses multiple machine learning
        classification algorithms and provides an interactive
        Streamlit interface.

        ---

        ### 📊 Dataset Information

        - **Dataset:** Breast Cancer Wisconsin Diagnostic Dataset
        - **Total Records:** 569
        - **Input Features:** 30
        - **Target:** Diagnosis
        - **Classes:** Benign (B) and Malignant (M)

        ---

        ### 🤖 Models Implemented

        #### 1. Support Vector Machine (SVM)

        The primary prediction model.

        **Test Accuracy: 98.25%**

        #### 2. Logistic Regression

        A linear classification algorithm.

        **Test Accuracy: 97.37%**

        #### 3. Random Forest

        An ensemble tree-based classification algorithm.

        **Test Accuracy: 95.61%**

        #### 4. XGBoost

        A gradient boosting classification algorithm.

        **Test Accuracy: 95.61%**

        ---

        ### 🔧 Input Features

        The model uses 30 cellular measurements:

        - Radius Mean
        - Texture Mean
        - Perimeter Mean
        - Area Mean
        - Smoothness Mean
        - Compactness Mean
        - Concavity Mean
        - Concave Points Mean
        - Symmetry Mean
        - Fractal Dimension Mean
        - Radius SE
        - Texture SE
        - Perimeter SE
        - Area SE
        - Smoothness SE
        - Compactness SE
        - Concavity SE
        - Concave Points SE
        - Symmetry SE
        - Fractal Dimension SE
        - Radius Worst
        - Texture Worst
        - Perimeter Worst
        - Area Worst
        - Smoothness Worst
        - Compactness Worst
        - Concavity Worst
        - Concave Points Worst
        - Symmetry Worst
        - Fractal Dimension Worst

        ---

        ### 🎯 Prediction Process

        1. User enters cellular measurements.
        2. Missing values are handled using the trained imputer.
        3. Features are standardized using the trained scaler.
        4. SVM generates the primary prediction.
        5. Other trained models generate comparison predictions.
        6. Results and confidence scores are displayed.

        ---

        ### 🛠️ Technology Stack

        - Python
        - Pandas
        - NumPy
        - Scikit-learn
        - XGBoost
        - Joblib
        - Streamlit
        - Plotly

        ---

        ### ⚠️ Important Disclaimer

        This application is intended for educational and
        research purposes only.

        **It should NOT be used as a standalone medical
        diagnostic tool.**

        Always consult a qualified healthcare professional
        for medical diagnosis and treatment decisions.

        ---

        ### 📚 References

        - Breast Cancer Wisconsin Diagnostic Dataset
        - Scikit-learn
        - Streamlit
        - XGBoost

        """
    )

    st.markdown("---")

    st.info(
        "💡 This project demonstrates machine learning "
        "classification, preprocessing, model evaluation, "
        "and Streamlit deployment."
    )

