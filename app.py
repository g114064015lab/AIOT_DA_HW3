import streamlit as st
import joblib
import os
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

def clean_text(text):
    """Clean and preprocess text"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

# Set page title and favicon
st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📱"
)

def train_model():
    """Train a new model if one doesn't exist"""
    
    # Use a public dataset URL
    url = "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Detection/master/spam.csv"
    try:
        # Download and prepare data
        df = pd.read_csv(url, encoding='latin-1')
        df['label'] = (df['v1'] == 'spam').astype(int)
        df['text'] = df['v2']
        
        # Create and train the model
        model = Pipeline([
            ('tfidf', TfidfVectorizer(
                strip_accents='unicode',
                ngram_range=(1, 2),
                stop_words='english',
                max_features=10000
            )),
            ('clf', MultinomialNB())
        ])
        
        # Train the model
        model.fit(df['text'], df['label'])
        
        # Save model
        model_path = os.path.join(os.getcwd(), 'models', 'final_model.pkl')
        joblib.dump(model, model_path)
        return model
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        st.error(f"Current working directory: {os.getcwd()}")
        st.error(f"Directory contents: {os.listdir()}")
        return None

def load_model():
    try:
        # Try to create models directory
        os.makedirs('models', exist_ok=True)
        
        model_path = os.path.join(os.getcwd(), 'models', 'final_model.pkl')
        if not os.path.exists(model_path):
            st.warning("⚠️ Model not found! Training a new model...")
            return train_model()
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return train_model()

def main():
    # Header with styling
    st.title("📱 SMS Spam Classifier")
    st.markdown("""
    This app helps you detect spam SMS messages using machine learning.
    Simply enter your message below to test it!
    """)
    
    # Load the model
    model = load_model()
    
    # Text input area
    message = st.text_area("Enter your SMS message:", height=100)
    
    if st.button("Classify Message", type="primary"):
        if not message:
            st.warning("⚠️ Please enter a message first!")
            return
        
        if model is None:
            return
            
        # Make prediction
        prediction = model.predict([message])[0]
        probability = model.predict_proba([message])[0]
        
        # Display result with styling
        st.markdown("---")
        st.subheader("Classification Result:")
        
        if prediction == 1:
            st.error("🚨 This message is likely SPAM!")
            confidence = probability[1] * 100
        else:
            st.success("✅ This message appears to be legitimate (HAM)")
            confidence = probability[0] * 100
            
        st.markdown(f"**Confidence**: {confidence:.2f}%")
        
        # Show cleaned text
        st.markdown("---")
        st.subheader("Preprocessed Text:")
        cleaned = clean_text(message)
        st.code(cleaned)
    
    # Add model info section
    with st.expander("ℹ️ Model Information"):
        if model is not None:
            st.markdown("""
            - **Model Type**: Multinomial Naive Bayes with TF-IDF
            - **Features**: Word bigrams
            - **Text Processing**: URL removal, case normalization, special character removal
            """)
        
    # Add footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Built with ❤️ using Streamlit | 
            <a href='https://github.com/g114064015lab/AIOT_DA_HW3'>View on GitHub</a></p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()