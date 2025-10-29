import pytest
from src.preprocess import clean_text, TextPreprocessor

def test_clean_text():
    # Test URL removal
    text = "Check this link http://spam.com and www.spam.com"
    cleaned = clean_text(text)
    assert "http" not in cleaned
    assert "www" not in cleaned
    
    # Test case normalization
    text = "HELLO World"
    cleaned = clean_text(text)
    assert cleaned == "hello world"
    
    # Test special character removal
    text = "Hello! @#$% World?"
    cleaned = clean_text(text)
    assert cleaned == "hello world"
    
def test_preprocessor():
    preprocessor = TextPreprocessor()
    
    # Test transform on single message
    text = "Hello! Click http://spam.com NOW!"
    features = preprocessor.transform([text])
    assert features.shape[1] > 0  # Should have some features
    
    # Test fit_transform
    texts = ["Hello world", "Spam spam spam", "Click here now"]
    features = preprocessor.fit_transform(texts)
    assert features.shape[0] == 3  # Should have 3 samples
    assert features.shape[1] > 0  # Should have some features