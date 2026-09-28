# ============================================================
# FAKE NEWS DETECTION SYSTEM
# Streamlit Application
#
# Research Approaches:
# 1. TF-IDF Only
# 2. TF-IDF + 6 Stylometric Features
# 3. Proposed Ensemble
# ============================================================


import streamlit as st
import joblib
import re
import string
import pandas as pd
import numpy as np

from scipy.sparse import hstack, csr_matrix


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="wide"
)


# ============================================================
# 2. LOAD SAVED MODELS AND PREPROCESSING OBJECTS
# ============================================================

@st.cache_resource
def load_models():

    # --------------------------------------------------------
    # TF-IDF vectorizer
    # --------------------------------------------------------

    tfidf = joblib.load(
        "saved_model/tfidf_vectorizer.pkl"
    )

    # --------------------------------------------------------
    # Stylometric feature scaler
    # --------------------------------------------------------

    scaler = joblib.load(
        "saved_model/stylometric_scaler.pkl"
    )

    # --------------------------------------------------------
    # Five individual models trained using
    # TF-IDF + 6 stylometric features
    # --------------------------------------------------------

    fused_models = joblib.load(
        "saved_model/fused_models.pkl"
    )

    # --------------------------------------------------------
    # Proposed ensemble
    # --------------------------------------------------------

    ensemble = joblib.load(
        "saved_model/proposed_ensemble.pkl"
    )

    # --------------------------------------------------------
    # Baseline SVM (TF-IDF Only)
    # --------------------------------------------------------

    baseline_svm = joblib.load(
        "saved_model/baseline_svm.pkl"
    )

    return tfidf, scaler, fused_models, ensemble, baseline_svm


tfidf, scaler, fused_models, ensemble, baseline_svm = load_models()


# ============================================================
# 3. TEXT CLEANING FUNCTION
# ============================================================

def clean_text(text):

    text = str(text)

    # --------------------------------------------------------
    # Remove Reuters patterns
    # --------------------------------------------------------

    text = re.sub(
        r"\b[A-Z][A-Za-z\s]*\(Reuters\)\s*-+\s*",
        "",
        text
    )

    text = re.sub(
        r"\(Reuters\)",
        "",
        text
    )

    # --------------------------------------------------------
    # Remove AP patterns
    # --------------------------------------------------------

    text = re.sub(
        r"\b[A-Z][A-Za-z\s]*\(AP\)\s*-+\s*",
        "",
        text
    )

    text = re.sub(
        r"\(AP\)",
        "",
        text
    )

    # --------------------------------------------------------
    # Convert to lowercase
    # --------------------------------------------------------

    text = text.lower()

    # --------------------------------------------------------
    # Remove URLs
    # --------------------------------------------------------

    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    # --------------------------------------------------------
    # Remove numbers
    # --------------------------------------------------------

    text = re.sub(
        r"\d+",
        "",
        text
    )

    # --------------------------------------------------------
    # Remove punctuation
    # --------------------------------------------------------

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # --------------------------------------------------------
    # Remove extra spaces
    # --------------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# 4. STYLOMETRIC FEATURE FUNCTION
# ============================================================

def stylometric_features(text):

    text = str(text)

    # Character count
    char_count = len(text)

    # Split text into words
    words = text.split()

    # Word count
    word_count = len(words)

    # Number of exclamation marks
    exclaim_count = text.count("!")

    # Number of question marks
    question_count = text.count("?")

    # Capital letter ratio
    capital_ratio = (
        sum(1 for c in text if c.isupper())
        / (char_count + 1)
    )

    # Average word length
    avg_word_length = (
        char_count
        / (word_count + 1)
    )

    # Return six stylometric features
    return pd.DataFrame([{

        "char_count": char_count,

        "word_count": word_count,

        "exclaim_count": exclaim_count,

        "question_count": question_count,

        "capital_ratio": capital_ratio,

        "avg_word_length": avg_word_length

    }])


# ============================================================
# 5. CREATE FUSED FEATURES
# ============================================================

def create_fused_features(news_text):

    # --------------------------------------------------------
    # Clean text for TF-IDF
    # --------------------------------------------------------

    cleaned_text = clean_text(news_text)

    # --------------------------------------------------------
    # Create TF-IDF features
    # --------------------------------------------------------

    tfidf_features = tfidf.transform(
        [cleaned_text]
    )

    # --------------------------------------------------------
    # Create six stylometric features
    # --------------------------------------------------------

    style_features = stylometric_features(
        news_text
    )

    # --------------------------------------------------------
    # Scale stylometric features
    # --------------------------------------------------------

    style_scaled = scaler.transform(
        style_features
    )

    # --------------------------------------------------------
    # Combine:
    # 5,000 TF-IDF + 6 stylometric = 5,006 features
    # --------------------------------------------------------

    fused_features = hstack([
        tfidf_features,
        csr_matrix(style_scaled)
    ])

    return tfidf_features, fused_features


# ============================================================
# 6. CALCULATE SVM CONFIDENCE
# ============================================================

def calculate_svm_confidence(model, features):

    """
    LinearSVC does not directly provide probabilities.

    Therefore, the decision_function score is converted into
    a confidence-like percentage.

    This should be interpreted as prediction confidence,
    NOT as a calibrated probability.
    """

    decision_score = model.decision_function(
        features
    )

    # Convert possible array output to a single value
    if isinstance(decision_score, np.ndarray):
        decision_score = float(
            np.ravel(decision_score)[0]
        )

    # Use absolute distance from the decision boundary.
    #
    # Larger distance = stronger model decision.
    confidence = (
        1 /
        (
            1 +
            np.exp(-abs(decision_score))
        )
    ) * 100

    # Keep within 50-100%
    confidence = max(
        50.0,
        min(99.99, confidence)
    )

    return confidence


# ============================================================
# 7. CALCULATE ENSEMBLE CONFIDENCE
# ============================================================

def calculate_ensemble_confidence(model, features):

    """
    Calculate confidence for the proposed ensemble.

    Priority:
    1. predict_proba()
    2. decision_function()
    3. voting agreement, if available
    4. otherwise None
    """

    # --------------------------------------------------------
    # Method 1: Probability
    # --------------------------------------------------------

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            features
        )[0]

        confidence = max(probabilities) * 100

        return confidence


    # --------------------------------------------------------
    # Method 2: Decision function
    # --------------------------------------------------------

    elif hasattr(model, "decision_function"):

        decision_score = model.decision_function(
            features
        )

        if isinstance(decision_score, np.ndarray):
            decision_score = float(
                np.ravel(decision_score)[0]
            )

        confidence = (
            1 /
            (
                1 +
                np.exp(-abs(decision_score))
            )
        ) * 100

        confidence = max(
            50.0,
            min(99.99, confidence)
        )

        return confidence


    # --------------------------------------------------------
    # Method 3: Voting agreement
    # --------------------------------------------------------

    elif hasattr(model, "estimators_"):

        try:

            predictions = []

            for estimator in model.estimators_:

                pred = estimator.predict(
                    features
                )[0]

                predictions.append(pred)

            if len(predictions) > 0:

                counts = pd.Series(
                    predictions
                ).value_counts()

                confidence = (
                    counts.iloc[0]
                    / len(predictions)
                ) * 100

                return confidence

        except Exception:
            pass


    # --------------------------------------------------------
    # If confidence cannot be calculated
    # --------------------------------------------------------

    return None


# ============================================================
# 8. APPLICATION TITLE
# ============================================================

st.title(
    "📰 Fake News Detection System"
)

st.write(
    """
    This application demonstrates the three research approaches
    investigated in the Fake News Detection System:

    **TF-IDF Only → TF-IDF + 6 Stylometric Features → Proposed Ensemble**
    """
)


# ============================================================
# 9. RESEARCH PERFORMANCE COMPARISON
# ============================================================

st.subheader(
    "📊 Research Approach Performance"
)

col1, col2, col3 = st.columns(3)


# ------------------------------------------------------------
# TF-IDF Only
# ------------------------------------------------------------

with col1:

    st.metric(
        "TF-IDF Only",
        "98.77%"
    )

    st.caption(
        "Best individual model: SVM"
    )


# ------------------------------------------------------------
# TF-IDF + 6 Features
# ------------------------------------------------------------

with col2:

    st.metric(
        "TF-IDF + 6 Features",
        "99.14%"
    )

    st.caption(
        "Best individual model: SVM"
    )


# ------------------------------------------------------------
# Proposed Ensemble
# ------------------------------------------------------------

with col3:

    st.metric(
        "Proposed Ensemble",
        "99.18%"
    )

    st.caption(
        "Proposed research approach"
    )


# ============================================================
# 10. BEST PERFORMING APPROACH
# ============================================================

st.info(
    "🏆 Based on the test accuracy reported in the thesis, "
    "the Proposed Ensemble achieved the highest accuracy: 99.08%."
)


st.divider()


# ============================================================
# 11. SELECT RESEARCH APPROACH
# ============================================================

st.subheader(
    "🔬 Select Research Approach"
)

approach = st.radio(
    "Choose the approach you want to use for prediction:",
    [
        "TF-IDF Only",
        "TF-IDF + 6 Stylometric Features",
        "Proposed Ensemble"
    ],
    horizontal=True
)


# ============================================================
# 12. SHOW SELECTED APPROACH INFORMATION
# ============================================================

if approach == "TF-IDF Only":

    st.write(
        "**Selected:** TF-IDF Only"
    )

    st.caption(
        "Uses 5,000 TF-IDF features."
    )

    selected_accuracy = "98.67%"


elif approach == "TF-IDF + 6 Stylometric Features":

    st.write(
        "**Selected:** TF-IDF + 6 Stylometric Features"
    )

    st.caption(
        "Uses 5,000 TF-IDF features + "
        "6 stylometric features = 5,006 features."
    )

    selected_accuracy = "99.04%"


else:

    st.write(
        "**Selected:** Proposed Ensemble"
    )

    st.caption(
        "Uses the proposed ensemble model with "
        "the fused 5,006-feature representation."
    )

    selected_accuracy = "99.08%"


# ============================================================
# 13. DISPLAY SELECTED APPROACH ACCURACY
# ============================================================

st.write(
    f"**Reported Test Accuracy:** {selected_accuracy}"
)


# ============================================================
# 14. NEWS ARTICLE INPUT
# ============================================================

st.subheader(
    "📝 Enter News Article"
)

news_text = st.text_area(
    "Paste the news article here:",
    height=300,
    placeholder="Paste the complete news article here..."
)


# ============================================================
# 15. PREDICTION BUTTON
# ============================================================

if st.button(
    "🔍 Predict News",
    type="primary"
):

    # --------------------------------------------------------
    # Check whether text was entered
    # --------------------------------------------------------

    if not news_text.strip():

        st.warning(
            "Please enter a news article before prediction."
        )

    else:

        # ----------------------------------------------------
        # CREATE FEATURES
        # ----------------------------------------------------

        tfidf_features, fused_features = (
            create_fused_features(news_text)
        )


        # ====================================================
        # VARIABLES FOR RESULT
        # ====================================================

        prediction = None

        selected_model_name = ""

        prediction_confidence = None


        # ====================================================
        # APPROACH 1: TF-IDF ONLY
        # ====================================================

        if approach == "TF-IDF Only":

            # ------------------------------------------------
            # Predict
            # ------------------------------------------------

            prediction = baseline_svm.predict(
                tfidf_features
            )[0]

            # ------------------------------------------------
            # Calculate prediction confidence
            # ------------------------------------------------

            prediction_confidence = (
                calculate_svm_confidence(
                    baseline_svm,
                    tfidf_features
                )
            )

            selected_model_name = (
                "SVM - TF-IDF Only"
            )


        # ====================================================
        # APPROACH 2:
        # TF-IDF + 6 STYLOMETRIC FEATURES
        # ====================================================

        elif approach == "TF-IDF + 6 Stylometric Features":

            # ------------------------------------------------
            # Get SVM from fused models
            # ------------------------------------------------

            fused_svm = fused_models["SVM"]

            # ------------------------------------------------
            # Predict
            # ------------------------------------------------

            prediction = fused_svm.predict(
                fused_features
            )[0]

            # ------------------------------------------------
            # Calculate prediction confidence
            # ------------------------------------------------

            prediction_confidence = (
                calculate_svm_confidence(
                    fused_svm,
                    fused_features
                )
            )

            selected_model_name = (
                "SVM - TF-IDF + 6 Stylometric Features"
            )


        # ====================================================
        # APPROACH 3: PROPOSED ENSEMBLE
        # ====================================================

        else:

            # ------------------------------------------------
            # Predict
            # ------------------------------------------------

            prediction = ensemble.predict(
                fused_features
            )[0]

            # ------------------------------------------------
            # Calculate ensemble confidence
            # ------------------------------------------------

            prediction_confidence = (
                calculate_ensemble_confidence(
                    ensemble,
                    fused_features
                )
            )

            selected_model_name = (
                "Proposed Ensemble"
            )


        # ====================================================
        # 16. DISPLAY PREDICTION RESULT
        # ====================================================

        st.divider()

        st.subheader(
            "🔎 Prediction Result"
        )


        # ----------------------------------------------------
        # FAKE NEWS
        # ----------------------------------------------------

        if prediction == 0:

            st.error(
                "🚨 FAKE NEWS"
            )

            st.write(
                "The selected research approach classified "
                "this article as **Fake News**."
            )


        # ----------------------------------------------------
        # REAL NEWS
        # ----------------------------------------------------

        elif prediction == 1:

            st.success(
                "✅ REAL NEWS"
            )

            st.write(
                "The selected research approach classified "
                "this article as **Real News**."
            )


        # ----------------------------------------------------
        # UNKNOWN LABEL
        # ----------------------------------------------------

        else:

            st.warning(
                f"Unexpected model output: {prediction}"
            )


        # ====================================================
        # 17. DISPLAY PREDICTION INFORMATION
        # ====================================================

        st.subheader(
            "📌 Prediction Information"
        )

        col1, col2, col3 = st.columns(3)


        # ----------------------------------------------------
        # Research Approach
        # ----------------------------------------------------

        with col1:

            st.write(
                "**Research Approach**"
            )

            st.write(
                approach
            )


        # ----------------------------------------------------
        # Model Used
        # ----------------------------------------------------

        with col2:

            st.write(
                "**Model Used**"
            )

            st.write(
                selected_model_name
            )


        # ----------------------------------------------------
        # Prediction Confidence
        # ----------------------------------------------------

        with col3:

            st.write(
                "**Prediction Confidence**"
            )

            if prediction_confidence is not None:

                st.metric(
                    "Confidence",
                    f"{prediction_confidence:.2f}%"
                )

            else:

                st.write(
                    "Not available"
                )


        # ====================================================
        # 18. REPORTED RESEARCH ACCURACY
        # ====================================================

        st.write("")

        st.write(
            "**Reported Test Accuracy:** "
            f"{selected_accuracy}"
        )

        st.caption(
            "Reported Test Accuracy represents the model's "
            "performance on the research test set. "
            "Prediction Confidence represents the model's "
            "confidence for the current article."
        )


# ============================================================
# 19. FEATURE INFORMATION
# ============================================================

st.divider()

st.subheader(
    "📚 Feature Information"
)

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "TF-IDF Features",
        "5,000"
    )


with col2:

    st.metric(
        "Stylometric Features",
        "6"
    )


with col3:

    st.metric(
        "Fused Features",
        "5,006"
    )


# ============================================================
# 20. FOOTER
# ============================================================

st.divider()

st.caption(
    "Fake News Detection System | "
    "Comparative Analysis of Machine Learning Algorithms"
)