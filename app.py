import streamlit as st
import joblib
import os
from src.preprocess import clean_text
import pandas as pd

# Set page title and favicon
st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📱"
)

def load_model():
    model_path = "models/final_model.pkl"
    if not os.path.exists(model_path):
        st.error("⚠️ Model not found! Please train the model first using main.py")
        return None
    return joblib.load(model_path)

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