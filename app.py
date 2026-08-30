"""
Customer Churn Prediction System
Streamlit Web Application
"""

import os
import sys
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import data_preprocessing
    import eda
    import predict
except ModuleNotFoundError as e:
    st.error(f"Project module could not be imported: {e}")
    st.stop()


STORAGE_DIR = os.path.join(ROOT_DIR, "models")

PIPELINE_PATH = os.path.join(
    STORAGE_DIR,
    "preprocessing_pipeline.pkl"
)

METADATA_PATH = os.path.join(
    STORAGE_DIR,
    "metadata.json"
)

DATA_PATH = os.path.join(
    ROOT_DIR,
    "data",
    "customer_churn.csv"
)

MODEL_PATHS = {
    "Random Forest": os.path.join(
        STORAGE_DIR,
        "random_forest.pkl"
    ),
    "Logistic Regression": os.path.join(
        STORAGE_DIR,
        "logistic_regression.pkl"
    ),
    "Decision Tree": os.path.join(
        STORAGE_DIR,
        "decision_tree.pkl"
    ),
    "Artificial Neural Network (ANN)": os.path.join(
        STORAGE_DIR,
        "churn_ann.keras"
    )
}


@st.cache_data
def load_churn_data():
    if not os.path.exists(DATA_PATH):
        return None

    try:
        return pd.read_csv(DATA_PATH)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None


@st.cache_data
def load_metadata():
    if not os.path.exists(METADATA_PATH):
        return None

    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as e:
        st.error(f"Error loading metadata: {e}")
        return None


@st.cache_resource
def load_pipeline_artifact():
    if not os.path.exists(PIPELINE_PATH):
        return None

    try:
        return predict.load_pipeline(PIPELINE_PATH)
    except Exception as e:
        st.error(f"Error loading preprocessing pipeline: {e}")
        return None


@st.cache_resource
def load_model_artifact(path):
    if not os.path.exists(path):
        return None

    try:
        return predict.load_model(path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def dashboard_page(df):
    st.title("Customer Churn Dashboard")
    st.write(
        "Real-time operational summary and key customer churn metrics."
    )

    if df is None:
        st.error(
            "Dataset customer_churn.csv was not found inside the data directory."
        )
        return

    try:
        df_clean = data_preprocessing.preprocess_dataframe(df)
    except Exception as e:
        st.error(f"Error during data preprocessing: {e}")
        return

    try:
        total_customers = len(df_clean)
        churned_customers = int(df_clean["Churn"].sum())
        churn_rate = (
            churned_customers / total_customers
        ) * 100

        avg_monthly = df_clean["MonthlyCharges"].mean()
        avg_tenure = df_clean["tenure"].mean()

    except Exception as e:
        st.error(f"Unable to calculate dashboard metrics: {e}")
        return

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "Churned Customers",
        f"{churned_customers:,}"
    )

    col3.metric(
        "Churn Rate",
        f"{churn_rate:.2f}%"
    )

    col4.metric(
        "Avg Monthly Charges",
        f"${avg_monthly:.2f}"
    )

    col5.metric(
        "Avg Tenure",
        f"{avg_tenure:.1f} mos"
    )

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Customer Churn Ratio")

        try:
            fig1 = eda.plot_churn_distribution(df)
            st.pyplot(fig1)
            plt.close(fig1)
        except Exception as e:
            st.error(f"Unable to create churn distribution: {e}")

    with c2:
        st.subheader("Churn Distribution by Contract Type")

        try:
            fig2 = eda.plot_churn_by_contract(df)
            st.pyplot(fig2)
            plt.close(fig2)
        except Exception as e:
            st.error(f"Unable to create contract chart: {e}")

    st.markdown("---")

    c3, c4 = st.columns(2)

    with c3:
        st.subheader("Tenure Distribution")

        try:
            fig3 = eda.plot_tenure_distribution(df)
            st.pyplot(fig3)
            plt.close(fig3)
        except Exception as e:
            st.error(f"Unable to create tenure chart: {e}")

    with c4:
        st.subheader("Monthly Charges Distribution")

        try:
            fig4 = eda.plot_monthly_charges_distribution(df)
            st.pyplot(fig4)
            plt.close(fig4)
        except Exception as e:
            st.error(f"Unable to create monthly charges chart: {e}")


def customer_analysis_page(df):
    st.title(
        "Exploratory Data Analysis and Statistical Summary"
    )

    if df is None:
        st.error("Dataset not found.")
        return

    st.subheader("1. Descriptive Statistics")

    try:
        stats_df = eda.get_descriptive_stats(df)

        st.dataframe(
            stats_df,
            use_container_width=True
        )

    except Exception as e:
        st.error(
            f"Unable to calculate descriptive statistics: {e}"
        )

    st.markdown("---")

    st.subheader("2. Exploratory Visualizations")

    selected_plot = st.selectbox(
        "Choose Visualization",
        [
            "1. Churn Overall Distribution",
            "2. Churn by Gender",
            "3. Churn by Contract Type",
            "4. Churn by Internet Service",
            "5. Churn by Payment Method",
            "6. Tenure Distribution",
            "7. Monthly Charges Distribution",
            "8. Correlation Heatmap"
        ]
    )

    fig = None
    insight = ""

    try:
        if "1. Churn Overall" in selected_plot:
            fig = eda.plot_churn_distribution(df)
            insight = (
                "Approximately 26.5% of customers have churned."
            )

        elif "2. Churn by Gender" in selected_plot:
            fig = eda.plot_churn_by_gender(df)
            insight = (
                "Churn is relatively balanced between male and female customers."
            )

        elif "3. Churn by Contract" in selected_plot:
            fig = eda.plot_churn_by_contract(df)
            insight = (
                "Month-to-month contract customers have considerably higher churn."
            )

        elif "4. Churn by Internet" in selected_plot:
            fig = eda.plot_churn_by_internet(df)
            insight = (
                "Fiber optic customers show relatively high churn."
            )

        elif "5. Churn by Payment" in selected_plot:
            fig = eda.plot_churn_by_payment(df)
            insight = (
                "Electronic check customers show higher churn."
            )

        elif "6. Tenure Distribution" in selected_plot:
            fig = eda.plot_tenure_distribution(df)
            insight = (
                "Customers with shorter tenure are generally at greater churn risk."
            )

        elif "7. Monthly Charges" in selected_plot:
            fig = eda.plot_monthly_charges_distribution(df)
            insight = (
                "Higher monthly charges are associated with increased churn."
            )

        elif "8. Correlation Heatmap" in selected_plot:
            fig = eda.plot_correlation_heatmap(df)
            insight = (
                "Tenure generally has a negative relationship with churn, "
                "while monthly charges show a positive relationship."
            )

    except Exception as e:
        st.error(f"Unable to create visualization: {e}")
        return

    if fig is not None:
        st.pyplot(fig)

        try:
            plt.close(fig)
        except Exception:
            pass

        st.caption(f"Insight: {insight}")


def churn_prediction_page(pipeline):
    st.title("Predict Customer Churn")

    st.write(
        "Enter customer information to generate a churn prediction "
        "and risk assessment."
    )

    if pipeline is None:
        st.error(
            "Preprocessing pipeline not found. Please train the models first."
        )
        return

    selected_model_name = st.selectbox(
        "Select ML or DL Model for Inference",
        list(MODEL_PATHS.keys())
    )

    model_path = MODEL_PATHS[selected_model_name]

    if not os.path.exists(model_path):
        st.warning(
            f"{selected_model_name} model has not been trained yet."
        )
        return

    model = load_model_artifact(model_path)

    if model is None:
        st.error(
            f"Unable to load {selected_model_name} model."
        )
        return

    with st.form("churn_prediction_form"):

        st.subheader(
            "Customer Demographics and Subscription Inputs"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            gender = st.selectbox(
                "Gender",
                ["Male", "Female"]
            )

            senior = st.selectbox(
                "Senior Citizen",
                [0, 1],
                format_func=lambda x:
                "Yes" if x == 1 else "No"
            )

            partner = st.selectbox(
                "Partner",
                ["Yes", "No"]
            )

            dependents = st.selectbox(
                "Dependents",
                ["Yes", "No"]
            )

            tenure = st.number_input(
                "Tenure Months",
                min_value=0,
                max_value=120,
                value=12
            )

            phone = st.selectbox(
                "Phone Service",
                ["Yes", "No"]
            )

            multiple = st.selectbox(
                "Multiple Lines",
                [
                    "No",
                    "Yes",
                    "No phone service"
                ]
            )

        with c2:

            internet = st.selectbox(
                "Internet Service",
                [
                    "DSL",
                    "Fiber optic",
                    "No"
                ]
            )

            security = st.selectbox(
                "Online Security",
                [
                    "No",
                    "Yes",
                    "No internet service"
                ]
            )

            backup = st.selectbox(
                "Online Backup",
                [
                    "No",
                    "Yes",
                    "No internet service"
                ]
            )

            device_prot = st.selectbox(
                "Device Protection",
                [
                    "No",
                    "Yes",
                    "No internet service"
                ]
            )

            tech_supp = st.selectbox(
                "Tech Support",
                [
                    "No",
                    "Yes",
                    "No internet service"
                ]
            )

            stream_tv = st.selectbox(
                "Streaming TV",
                [
                    "No",
                    "Yes",
                    "No internet service"
                ]
            )

        with c3:

            stream_mov = st.selectbox(
                "Streaming Movies",
                [
                    "No",
                    "Yes",
                    "No internet service"
                ]
            )

            contract = st.selectbox(
                "Contract",
                [
                    "Month-to-month",
                    "One year",
                    "Two year"
                ]
            )

            paperless = st.selectbox(
                "Paperless Billing",
                [
                    "Yes",
                    "No"
                ]
            )

            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

            monthly = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                max_value=500.0,
                value=70.35,
                step=1.0
            )

            total = st.number_input(
                "Total Charges",
                min_value=0.0,
                max_value=20000.0,
                value=float(
                    round(monthly * tenure, 2)
                ),
                step=10.0
            )

        submitted = st.form_submit_button(
            "PREDICT CHURN",
            use_container_width=True
        )

    if submitted:

        input_dict = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": device_prot,
            "TechSupport": tech_supp,
            "StreamingTV": stream_tv,
            "StreamingMovies": stream_mov,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total
        }

        try:
            with st.spinner(
                "Analyzing customer profile..."
            ):
                result = predict.predict_single(
                    input_dict,
                    pipeline,
                    model
                )

        except Exception as e:
            st.error(
                f"Prediction failed: {e}"
            )
            return

        st.markdown("---")
        st.subheader("Prediction Results")

        r1, r2, r3 = st.columns(3)

        with r1:

            if result["is_churn"]:
                st.error(
                    f"Prediction\n\n"
                    f"{result['prediction_text']}"
                )
            else:
                st.success(
                    f"Prediction\n\n"
                    f"{result['prediction_text']}"
                )

        with r2:

            st.metric(
                "Churn Probability",
                f"{result['probability_percent']}%"
            )

        with r3:

            risk = result["risk_level"]

            if risk == "High Risk":
                st.error(
                    f"Risk Level\n\n{risk}"
                )

            elif risk == "Medium Risk":
                st.warning(
                    f"Risk Level\n\n{risk}"
                )

            else:
                st.success(
                    f"Risk Level\n\n{risk}"
                )


def model_performance_page(metadata):
    st.title("Model Evaluation and Comparison")

    if metadata is None:
        st.warning(
            "Model evaluation metadata was not found."
        )
        return

    if "models" not in metadata:
        st.warning(
            "models section is missing from metadata.json."
        )
        return

    models_eval = metadata["models"]

    model_names_map = {
        "logistic_regression": "Logistic Regression",
        "decision_tree": "Decision Tree",
        "random_forest": "Random Forest",
        "ann": "Artificial Neural Network (ANN)"
    }

    table_rows = []

    for key, value in models_eval.items():

        name = model_names_map.get(
            key,
            key.replace("_", " ").title()
        )

        table_rows.append(
            {
                "Model": name,
                "Accuracy": f"{value.get('accuracy', 0):.4f}",
                "Precision": f"{value.get('precision', 0):.4f}",
                "Recall": f"{value.get('recall', 0):.4f}",
                "F1 Score": f"{value.get('f1', 0):.4f}",
                "ROC AUC": f"{value.get('roc_auc', 0):.4f}"
            }
        )

    comp_df = pd.DataFrame(table_rows)

    st.subheader("1. Model Comparison Table")

    st.dataframe(
        comp_df,
        use_container_width=True
    )

    st.info(
        "F1 Score and ROC AUC are important for churn prediction "
        "because the goal is to identify potential churners while "
        "reducing false alarms."
    )

    st.markdown("---")

    st.subheader("2. ROC Curves Comparison")

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    has_roc = False

    for key, value in models_eval.items():

        name = model_names_map.get(
            key,
            key.replace("_", " ").title()
        )

        if "fpr" in value and "tpr" in value:

            ax.plot(
                value["fpr"],
                value["tpr"],
                label=(
                    f"{name} "
                    f"(AUC = "
                    f"{value.get('roc_auc', 0):.3f})"
                )
            )

            has_roc = True

    ax.plot(
        [0, 1],
        [0, 1],
        "k--",
        label="Random Chance"
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    ax.set_title(
        "ROC Curves Comparison",
        fontweight="bold"
    )

    if has_roc:
        ax.legend(
            loc="lower right"
        )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.markdown("---")

    st.subheader("3. Confusion Matrices")

    cols = st.columns(
        max(len(models_eval), 1)
    )

    for index, (key, value) in enumerate(
        models_eval.items()
    ):

        name = model_names_map.get(
            key,
            key.replace("_", " ").title()
        )

        with cols[index]:

            st.caption(name)

            if "confusion_matrix" not in value:
                st.warning(
                    "Confusion matrix unavailable."
                )
                continue

            cm = np.array(
                value["confusion_matrix"]
            )

            fig_cm, ax_cm = plt.subplots(
                figsize=(4, 3)
            )

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                ax=ax_cm
            )

            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")

            plt.tight_layout()

            st.pyplot(fig_cm)

            plt.close(fig_cm)

    if "ann_history" in metadata:

        st.markdown("---")

        st.subheader(
            "4. ANN Training vs Validation Curves"
        )

        history = metadata["ann_history"]

        fig_ann, (ax1, ax2) = plt.subplots(
            1,
            2,
            figsize=(12, 4)
        )

        ax1.plot(
            history.get("loss", []),
            label="Train Loss"
        )

        ax1.plot(
            history.get("val_loss", []),
            label="Validation Loss"
        )

        ax1.set_title("Loss Curves")
        ax1.set_xlabel("Epochs")
        ax1.legend()

        ax2.plot(
            history.get("accuracy", []),
            label="Train Accuracy"
        )

        ax2.plot(
            history.get("val_accuracy", []),
            label="Validation Accuracy"
        )

        ax2.set_title("Accuracy Curves")
        ax2.set_xlabel("Epochs")
        ax2.legend()

        plt.tight_layout()

        st.pyplot(fig_ann)

        plt.close(fig_ann)


def about_project_page():

    st.title(
        "About Customer Churn Prediction System"
    )

    st.markdown(
        """
        ### Project Architecture and Tech Stack

        This project demonstrates an end-to-end Data Science
        and Machine Learning pipeline using Python.

        ### Technologies

        - Python
        - NumPy
        - Pandas
        - Matplotlib
        - Seaborn
        - Scikit-Learn
        - TensorFlow and Keras
        - Streamlit

        ---

        ### Machine Learning Models

        The system compares:

        1. Logistic Regression
        2. Decision Tree
        3. Random Forest
        4. Artificial Neural Network

        ---

        ### Preprocessing

        - Missing value handling
        - Median imputation for numerical features
        - Mode imputation for categorical features
        - One Hot Encoding
        - Standard Scaling
        - Stratified train test split

        ---

        ### Data Leakage Prevention

        The preprocessing pipeline is fitted only on training
        data to avoid data leakage.

        ---

        ### Evaluation Metrics

        Models are evaluated using:

        - Accuracy
        - Precision
        - Recall
        - F1 Score
        - ROC AUC

        F1 Score and ROC AUC are particularly useful for churn
        prediction because correctly identifying customers
        likely to churn is important.

        ---

        ### Application Features

        - Interactive dashboard
        - Exploratory data analysis
        - Individual customer churn prediction
        - Model comparison
        - ROC curves
        - Confusion matrices
        - ANN training curves
        """
    )


def main():

    st.set_page_config(
        page_title="Customer Churn Prediction System",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.sidebar.title("Navigation")

    page = st.sidebar.radio(
        "Select Page",
        [
            "Dashboard",
            "Customer Analysis",
            "Churn Prediction",
            "Model Performance",
            "About Project"
        ]
    )

    df = load_churn_data()
    pipeline = load_pipeline_artifact()
    metadata = load_metadata()

    if page == "Dashboard":
        dashboard_page(df)

    elif page == "Customer Analysis":
        customer_analysis_page(df)

    elif page == "Churn Prediction":
        churn_prediction_page(pipeline)

    elif page == "Model Performance":
        model_performance_page(metadata)

    elif page == "About Project":
        about_project_page()


if __name__ == "__main__":
    main()