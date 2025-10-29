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

def get_fallback_dataset():
    """Return a small built-in dataset as fallback"""
    texts = [
        ("ham", "Hi honey, how are you? Call me when you get this."),
        ("spam", "WINNER!! You have won £1000, call now to claim!"),
        ("ham", "Meeting at 3pm in the conference room"),
        ("spam", "FREE entry into our £250 weekly competition"),
        ("ham", "Can you pick up milk on your way home?"),
        ("spam", "URGENT! Your bank account has been suspended"),
        ("ham", "Great seeing you yesterday, thanks for dinner"),
        ("spam", "Get 50% off designer watches! Limited time offer"),
        ("ham", "The report is ready for review"),
        ("spam", "You've won a free iPhone! Click here to claim"),
    ]
    return pd.DataFrame(texts, columns=['label', 'text'])

def train_model():
    """Train a new model if one doesn't exist"""
    try:
        # Try using built-in fallback dataset first
        st.info("Using built-in dataset for training")
        df = get_fallback_dataset()
            
        # Convert labels to binary format
        df['label'] = (df['label'].str.lower() == 'spam').astype(int)
        st.info(f"Number of spam messages: {df['label'].sum()}")
        st.info(f"Number of ham messages: {(df['label'] == 0).sum()}")
        
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
        model_dir = os.path.join(os.getcwd(), 'models')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'final_model.pkl')
        joblib.dump(model, model_path)
        
        st.success("✅ Model trained successfully!")
        return model
        
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        st.error(f"Current working directory: {os.getcwd()}")
        st.error(f"Directory contents: {os.listdir()}")
        return None

def load_model():
    """Load or train the model"""
    try:
        model_dir = os.path.join(os.getcwd(), 'models')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'final_model.pkl')
        
        if not os.path.exists(model_path):
            st.warning("⚠️ Model not found! Training a new model...")
            return train_model()
        
        model = joblib.load(model_path)
        st.success("✅ Model loaded successfully!")
        return model
        
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
            st.error("❌ Could not load or train model!")
            return
        
        try:
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
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    
    # Add model info section
    with st.expander("ℹ️ Model Information"):
        if model is not None:
            st.markdown("""
            - **Model Type**: Multinomial Naive Bayes with TF-IDF
            - **Features**: Word bigrams
            - **Text Processing**: URL removal, case normalization, special character removal
            - **Training Data**: Built-in dataset with spam/ham examples
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