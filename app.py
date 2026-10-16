
import streamlit as st
import joblib
import json
import os

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- CUSTOM UI --------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f5f7ff, #eef2ff);
    color: #17213b;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #10172f, #202b50);
}

[data-testid="stSidebar"] * {
    color: #f8faff;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #e1e7ff;
}

h1, h2, h3, h4 {
    color: #17213b;
}

.hero {
    background: linear-gradient(120deg, #ffffff, #f0edff);
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #ddd6fe;
    box-shadow: 0 6px 20px rgba(60, 50, 120, 0.08);
    margin-bottom: 20px;
}

div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e0e7ff;
    padding: 18px;
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(40, 50, 100, 0.05);
}

div[data-testid="stMetricLabel"] {
    color: #596580;
}

div[data-testid="stMetricValue"] {
    color: #5145cd;
    font-weight: 700;
}

.stTextArea textarea {
    background-color: #ffffff !important;
    color: #17213b !important;
    border: 1.5px solid #c4b5fd !important;
    border-radius: 12px !important;
    font-size: 16px !important;
}

.stTextArea textarea:focus {
    border: 2px solid #6d5ce7 !important;
    box-shadow: 0 0 0 2px #ede9fe !important;
}

div.stButton > button {
    background: linear-gradient(90deg, #5946df, #7865f5);
    color: #ffffff;
    border: none;
    border-radius: 11px;
    padding: 11px 18px;
    font-weight: 600;
    min-height: 44px;
    transition: 0.2s;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #4333bd, #6452e8);
    color: #ffffff;
    border: 1px solid #8b80ff;
    transform: translateY(-1px);
}

div.stButton > button[kind="secondary"] {
    background: #ffffff;
    color: #5946df;
    border: 1.5px solid #c4b5fd;
}

div.stButton > button[kind="secondary"]:hover {
    background: #ede9fe;
    color: #4338ca;
}

[data-testid="stProgress"] > div {
    background-color: #e5e7eb;
    border-radius: 10px;
}

[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #6d5ce7, #8b5cf6);
    border-radius: 10px;
}

hr {
    border-color: #dbe2f4;
}

.footer {
    text-align: center;
    color: #718096;
    font-size: 13px;
    padding: 15px;
}

.small-note {
    color: #64748b;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)


# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer


model, vectorizer = load_model()

try:
    with open("metrics.json", "r") as f:
        metrics = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    metrics = {}


# -------------------- SESSION STATE --------------------
if "news_input" not in st.session_state:
    st.session_state.news_input = ""

if "last_news" not in st.session_state:
    st.session_state.last_news = ""

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "last_probabilities" not in st.session_state:
    st.session_state.last_probabilities = None


def clear_news():
    st.session_state.news_input = ""
    st.session_state.last_news = ""
    st.session_state.last_prediction = None
    st.session_state.last_probabilities = None


# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.markdown("# 🛡️ Fake News")
    st.markdown("### Detection System")
    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Home",
            "🔎 Check News",
            "📊 Analytics",
            "ℹ️ About Project"
        ],
        label_visibility="visible"
    )

    st.markdown("---")
    st.markdown("### 💡 About")
    st.write(
        "An ML-powered system using TF-IDF and "
        "Multinomial Naive Bayes to classify news articles."
    )

    st.markdown("---")
    st.caption("Academic Project • 2026")


# -------------------- PREDICTION RESULT --------------------
def show_result():
    prediction = st.session_state.last_prediction
    probabilities = st.session_state.last_probabilities

    if prediction is None or probabilities is None:
        st.info("💡 Enter a news article and click Check News to see the prediction.")
        return

    confidence = probabilities.get(prediction, 0.0) * 100

    st.markdown("### 📌 Prediction Result")

    result_col, probability_col = st.columns([1.1, 1])

    with result_col:
        if prediction == 0:
            st.error("🚨 Prediction: Fake News")
            st.markdown(
                "**Model classification:** Label 0"
            )
        elif prediction == 1:
            st.success("✅ Prediction: Real News")
            st.markdown(
                "**Model classification:** Label 1"
            )

        st.metric(
            "Model Class Probability",
            f"{confidence:.2f}%"
        )

    with probability_col:
        st.markdown("#### 📊 Class Probabilities")

        fake_probability = probabilities.get(0, 0.0)
        real_probability = probabilities.get(1, 0.0)

        st.markdown("🔴 **Fake News (0)**")
        st.progress(float(fake_probability))
        st.caption(f"{fake_probability * 100:.2f}%")

        st.markdown("🟢 **Real News (1)**")
        st.progress(float(real_probability))
        st.caption(f"{real_probability * 100:.2f}%")

    st.caption(
        "Probability is the model's estimated class probability, "
        "not a guarantee that the news is factually true or false."
    )


# -------------------- NEWS CHECKER --------------------
def news_checker():
    st.markdown("""
    <div class="hero">
        <h2>🔎 Check a News Article</h2>
        <p>Paste the article below and let the ML model analyze its text.</p>
    </div>
    """, unsafe_allow_html=True)

    news = st.text_area(
        "📰 News Content",
        key="news_input",
        height=220,
        placeholder="Type or paste a news article here..."
    )

    check_col, clear_col = st.columns([1, 1])

    with check_col:
        check_clicked = st.button(
            "🔍 Check News",
            key="check_news",
            use_container_width=True
        )

    with clear_col:
        st.button(
            "🗑️ Clear",
            key="clear_news",
            type="secondary",
            use_container_width=True,
            on_click=clear_news
        )

    if check_clicked:
        if not news.strip():
            st.warning("Please enter some news content.")
        else:
            with st.spinner("Analyzing news article..."):
                features = vectorizer.transform([news])
                prediction = int(model.predict(features)[0])
                raw_probabilities = model.predict_proba(features)[0]

                probabilities = {
                    int(label): float(probability)
                    for label, probability in zip(
                        model.classes_,
                        raw_probabilities
                    )
                }

                st.session_state.last_news = news
                st.session_state.last_prediction = prediction
                st.session_state.last_probabilities = probabilities

    show_result()


# -------------------- HOME PAGE --------------------
if page == "🏠 Home":
    st.markdown("# Welcome! 👋")
    st.write("Your intelligent news analysis dashboard.")

    st.markdown("""
    <div class="hero">
        <h2>📰 Fake News Detection Dashboard</h2>
        <p>Analyze news articles using Machine Learning, explore model
        performance, and view prediction probabilities in one place.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📈 Model Performance Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "🎯 Accuracy",
        f"{metrics.get('accuracy', 0) * 100:.2f}%"
    )
    c2.metric(
        "🔍 Precision",
        f"{metrics.get('precision', 0) * 100:.2f}%"
    )
    c3.metric(
        "📌 Recall",
        f"{metrics.get('recall', 0) * 100:.2f}%"
    )
    c4.metric(
        "⚖️ F1-score",
        f"{metrics.get('f1_score', 0) * 100:.2f}%"
    )

    st.markdown("### 🧪 Try the News Detector")
    news_checker()


# -------------------- CHECK NEWS PAGE --------------------
elif page == "🔎 Check News":
    st.markdown("# 🔎 News Verification")
    st.write("Enter the full news text for classification.")
    news_checker()


# -------------------- ANALYTICS PAGE --------------------
elif page == "📊 Analytics":
    st.markdown("# 📊 Model Analytics")
    st.write("Evaluation results from the held-out test dataset.")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Accuracy",
        f"{metrics.get('accuracy', 0) * 100:.2f}%"
    )
    c2.metric(
        "Precision",
        f"{metrics.get('precision', 0) * 100:.2f}%"
    )
    c3.metric(
        "Recall",
        f"{metrics.get('recall', 0) * 100:.2f}%"
    )
    c4.metric(
        "F1-score",
        f"{metrics.get('f1_score', 0) * 100:.2f}%"
    )

    left, right = st.columns(2)

    with left:
        st.markdown("### 🧩 Confusion Matrix")

        if os.path.exists("confusion_matrix.png"):
            st.image(
                "confusion_matrix.png",
                use_container_width=True
            )
        else:
            st.warning("confusion_matrix.png not found.")

    with right:
        st.markdown("### 📊 Performance Graph")

        if os.path.exists("performance_graph.png"):
            st.image(
                "performance_graph.png",
                use_container_width=True
            )
        else:
            st.warning("performance_graph.png not found.")

    st.markdown("### ⚙️ Model Details")

    detail1, detail2 = st.columns(2)

    with detail1:
        st.markdown("**Algorithm**")
        st.write("Multinomial Naive Bayes")
        st.markdown("**Feature Extraction**")
        st.write("TF-IDF Vectorizer")

    with detail2:
        st.markdown("**N-grams**")
        st.write("Unigrams and bigrams")
        st.markdown("**Dataset**")
        st.write("WELFake_ML.csv")


# -------------------- ABOUT PAGE --------------------
elif page == "ℹ️ About Project":
    st.markdown("# ℹ️ About the Project")

    st.markdown("""
    <div class="hero">
        <h2>Fake News Detection Using Machine Learning</h2>
        <p>This academic project uses Natural Language Processing
        and Machine Learning to classify news text into dataset labels.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🔄 Project Workflow")

    st.write("1. Load and clean the news dataset.")
    st.write("2. Convert text into numerical features using TF-IDF.")
    st.write("3. Train the Multinomial Naive Bayes classifier.")
    st.write("4. Evaluate the model using Accuracy, Precision, Recall and F1-score.")
    st.write("5. Predict a label for new article text.")

    st.markdown("### 🧰 Technologies Used")

    st.write(
        "Python • Pandas • Scikit-learn • Streamlit • "
        "Joblib • Matplotlib"
    )

    st.markdown("### ⚠️ Important Note")

    st.info(
        "This system predicts labels learned from its training dataset. "
        "It does not independently verify news against trusted sources. "
        "Check reliable sources before deciding whether a claim is true."
    )


# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown(
    '<div class="footer">'
    '🛡️ Fake News Detection System | Academic Machine Learning Project'
    '</div>',
    unsafe_allow_html=True
)
