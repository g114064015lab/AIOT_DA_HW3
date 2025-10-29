"""Streamlit app for SMS spam classification."""
import os
from io import StringIO
import joblib
import streamlit as st
import pandas as pd
import re
import string
import json
import requests
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from typing import Optional, Tuple, Dict, Any

# Optional plotting libs
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except Exception:
    PLOTTING_AVAILABLE = False

# Default dataset URL from the project
DEFAULT_DATASET_URL = (
    "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/"
    "refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"
)

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Replace numbers with token
    text = re.sub(r'\d+', ' <NUM> ', text)
    
    # Remove punctuation but keep word boundaries
    text = re.sub(r'[^\w\s<>]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

@st.cache_data  # Cache the dataset loading
def load_default_dataset() -> Optional[pd.DataFrame]:
    """Load dataset from default URL if not available locally."""
    try:
        st.info("📥 Downloading default dataset...")
        response = requests.get(DEFAULT_DATASET_URL)
        response.raise_for_status()
        data = response.content.decode('utf-8')
        df = pd.read_csv(StringIO(data), header=None, names=['label', 'text'])
        df['label'] = df['label'].astype(str).str.strip().str.lower()
        df = df[df['label'].isin(['ham', 'spam'])].copy()
        df['text'] = df['text'].fillna('').astype(str)
        st.success("✅ Default dataset loaded successfully!")
        return df
    except Exception as e:
        st.error(f"Could not load default dataset: {str(e)}")
        return None

def load_dataset(file_path: Optional[str] = None, 
                uploaded_file: Optional[st.runtime.uploaded_file_manager.UploadedFile] = None) -> Tuple[Optional[pd.DataFrame], str]:
    """Load dataset from file path or uploaded file, return (dataframe, error_message)"""
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, header=None, names=['label', 'text'], encoding='utf-8')
        elif file_path and os.path.exists(file_path):
            df = pd.read_csv(file_path, header=None, names=['label', 'text'], encoding='utf-8')
        else:
            # Try loading default dataset
            df = load_default_dataset()
            if df is None:
                return None, f"Dataset not found at {file_path}. Please upload a CSV file or check the path."
            return df, ""
        
        # Validate and clean
        df['label'] = df['label'].astype(str).str.strip().str.lower()
        df = df[df['label'].isin(['ham', 'spam'])].copy()
        df['text'] = df['text'].fillna('').astype(str)
        
        if len(df) == 0:
            return None, "Dataset is empty or contains no valid ham/spam labels"
            
        return df, ""
        
    except Exception as e:
        return None, f"Error loading dataset: {str(e)}"

def load_or_train_model(model_path: str, 
                       dataset_path: Optional[str] = None,
                       uploaded_dataset: Optional[st.runtime.uploaded_file_manager.UploadedFile] = None,
                       force_retrain: bool = False,
                       random_state: int = 42) -> Tuple[Optional[Pipeline], str]:
    """Load saved model or train a new one if needed."""
    if not force_retrain and os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            return model, ""
        except Exception as e:
            return None, f"Error loading model: {str(e)}"
    
    # Load dataset for training
    df, error = load_dataset(dataset_path, uploaded_dataset)
    if df is None:
        return None, error
    
    try:
        # Quick baseline training
        from sklearn.model_selection import train_test_split
        from src.train import train_and_evaluate
        
        X = df['text'].values
        y = df['label'].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=random_state
        )
        
        pipeline, metrics = train_and_evaluate(
            X_train, X_test, y_train, y_test,
            model_type='lr',  # LogisticRegression with better defaults
            do_grid_search=False,
            random_state=random_state
        )
        
        # Save model and metrics
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(pipeline, model_path)
        
        metrics_path = os.path.join(os.path.dirname(model_path), 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)
            
        return pipeline, ""
        
    except Exception as e:
        return None, f"Error training model: {str(e)}"

# Page config and title
st.set_page_config(page_title="SMS Spam Classifier", page_icon="📱", layout="wide")

# Sidebar configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # Dataset settings
    st.header("Dataset")
    dataset_source = st.radio(
        "Choose dataset source:",
        ["Default path", "Upload CSV", "Custom path"]
    )
    
    if dataset_source == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload a CSV file (2 columns: label, text)",
            type="csv"
        )
    elif dataset_source == "Custom path":
        custom_path = st.text_input(
            "Dataset path:",
            value="data/sms_spam_no_header.csv"
        )
    else:
        custom_path = "data/sms_spam_no_header.csv"
        uploaded_file = None
    
    # Model settings
    st.header("Model")
    decision_threshold = st.slider(
        "Decision threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        help="Probability threshold for spam classification"
    )
    
    random_seed = st.number_input(
        "Random seed",
        value=42,
        help="Random seed for reproducibility"
    )
    
    force_retrain = st.checkbox(
        "Force retrain",
        value=False,
        help="Retrain model even if saved model exists"
    )

# Main content
st.title("📱 SMS Spam Classifier")
st.write(
    "This app uses machine learning to classify SMS messages as spam or ham (not spam). "
    "Enter a message below to try it out!"
)

# Load or train model based on configuration
dataset_path = None if dataset_source == "Upload CSV" else custom_path
model_path = os.path.join(os.getcwd(), "models", "final_model.pkl")
model, error = load_or_train_model(
    model_path=model_path,
    dataset_path=dataset_path,
    uploaded_dataset=uploaded_file if dataset_source == "Upload CSV" else None,
    force_retrain=force_retrain,
    random_state=random_seed
)

if error:
    st.error(error)
    if "Dataset not found" in error:
        st.info(
            "You can:\n"
            "1. Upload a CSV file using the sidebar\n"
            "2. Run training script: `python main.py`\n"
            "3. Download default dataset:\n"
            f"   - URL: {DEFAULT_DATASET_URL}\n"
            "   - Save to: data/sms_spam_no_header.csv"
        )

# Message input and prediction
col1, col2 = st.columns([3, 1])
with col1:
    message = st.text_area("Enter a message to classify:", value="", key="message_input", height=100)
with col2:
    st.write("")  # Add spacing
    detect_button = st.button("🔍 Detect Spam", type="primary", use_container_width=True)
    if detect_button:
        st.write("")  # Add spacing

if detect_button and message and model is not None:
    with st.spinner("Analyzing message..."):
        # Clean the message
        cleaned = clean_text(message)
        
        # Make prediction with threshold
        y_prob = model.predict_proba([cleaned])[0]
        is_spam = y_prob[1] >= decision_threshold
        
        # Show prediction with probability
        prob_spam = y_prob[1]
        
        if is_spam:
            st.error(f"🚨 Spam detected! (confidence: {prob_spam:.1%})")
        else:
            st.success(f"✅ Message looks legitimate (confidence: {1-prob_spam:.1%})")
        
        # Probability bars
        col1, col2 = st.columns(2)
        with col1:
            st.write("Ham probability:")
            st.progress(float(y_prob[0]))
        with col2:
            st.write("Spam probability:")
            st.progress(float(y_prob[1]))
        
        # Show preprocessing steps
        with st.expander("🔍 View text preprocessing steps", expanded=True):
            st.write("Original:", message)
            st.write("Cleaned:", cleaned)

# Dataset analysis - always expanded
with st.expander("📊 Dataset Analysis", expanded=True):
    # Try to get dataset from various sources
    df = None
    if dataset_source == "Upload CSV" and uploaded_file is not None:
        df = pd.read_csv(uploaded_file, header=None, names=['label', 'text'])
    elif os.path.exists(dataset_path or ""):
        df = pd.read_csv(dataset_path, header=None, names=['label', 'text'])
    
    # If no dataset found, try loading default
    if df is None:
        df = load_default_dataset()
        
    if df is not None:
        st.write(f"Total messages: {len(df)}")
        spam_count = (df['label'].str.lower() == 'spam').sum()
        ham_count = (df['label'].str.lower() == 'ham').sum()
        
        # Distribution plot
        if PLOTTING_AVAILABLE:
            fig, ax = plt.subplots()
            data = pd.DataFrame({
                'Label': ['Ham', 'Spam'],
                'Count': [ham_count, spam_count]
            })
            sns.barplot(x='Label', y='Count', data=data)
            plt.title('Message Distribution')
            st.pyplot(fig)
        else:
            st.write(f"Ham messages: {ham_count}")
            st.write(f"Spam messages: {spam_count}")
        
        # Word frequency analysis (always show)
        st.write("### Word Frequencies")
        with st.spinner("Computing word frequencies..."):
            # Get clean texts
            texts = df['text'].fillna('').astype(str).map(clean_text)
            
            # Compute word frequencies
            words = []
            for text in texts:
                words.extend(text.split())
            
            # Remove stopwords
            stopwords = set(ENGLISH_STOP_WORDS)
            word_freq = Counter(w for w in words if w not in stopwords)
            
            # Plot top words
            top_n = 20
            common_words = word_freq.most_common(top_n)
            
            if PLOTTING_AVAILABLE:
                fig, ax = plt.subplots(figsize=(10, 6))
                words, counts = zip(*common_words)
                sns.barplot(x=list(counts), y=list(words))
                plt.title(f'Top {top_n} Words')
                st.pyplot(fig)
            else:
                st.write(f"Top {top_n} words:")
                for word, count in common_words:
                    st.write(f"- {word}: {count}")
            
            # Add per-class word frequencies
            st.write("### Word frequencies by class")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Top words in spam messages:")
                spam_texts = texts[df['label'] == 'spam']
                spam_words = []
                for text in spam_texts:
                    spam_words.extend(text.split())
                spam_freq = Counter(w for w in spam_words if w not in stopwords)
                spam_common = spam_freq.most_common(10)
                
                if PLOTTING_AVAILABLE:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    words, counts = zip(*spam_common)
                    sns.barplot(x=list(counts), y=list(words))
                    plt.title('Top Spam Words')
                    st.pyplot(fig)
                else:
                    for word, count in spam_common:
                        st.write(f"- {word}: {count}")
                        
            with col2:
                st.write("Top words in ham messages:")
                ham_texts = texts[df['label'] == 'ham']
                ham_words = []
                for text in ham_texts:
                    ham_words.extend(text.split())
                ham_freq = Counter(w for w in ham_words if w not in stopwords)
                ham_common = ham_freq.most_common(10)
                
                if PLOTTING_AVAILABLE:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    words, counts = zip(*ham_common)
                    sns.barplot(x=list(counts), y=list(words))
                    plt.title('Top Ham Words')
                    st.pyplot(fig)
                else:
                    for word, count in ham_common:
                        st.write(f"- {word}: {count}")

# Model information and dataset analysis - always expanded
with st.expander("ℹ️ Model Information & Analysis", expanded=True):
    if model is not None:
        st.write("### Model Configuration")
        st.markdown(f"""
        - **Model Type**: Logistic Regression with TF-IDF
        - **Features**: Word bigrams
        - **Text Processing**: URL removal, case normalization, special character removal
        - **Settings**:
            - Decision threshold: {decision_threshold:.2f}
            - Random seed: {random_seed}
            - Dataset: {"Uploaded file" if dataset_source == "Upload CSV" else dataset_path}
        """)
