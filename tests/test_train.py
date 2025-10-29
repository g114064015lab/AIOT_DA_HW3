import pytest
import pandas as pd
from src.train import train_model, evaluate_model
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

@pytest.fixture
def sample_data():
    texts = [
        "Free money now", 
        "Click here to win", 
        "Meeting at 3pm",
        "Call me back please"
    ]
    labels = [1, 1, 0, 0]  # 1=spam, 0=ham
    return pd.DataFrame({'text': texts, 'label': labels})

def test_train_model(sample_data):
    # Test model training
    model = train_model(sample_data['text'], sample_data['label'])
    assert isinstance(model, Pipeline)
    
    # Test predictions
    preds = model.predict(["Free money"])
    assert len(preds) == 1
    assert preds[0] in [0, 1]
    
def test_evaluate_model(sample_data):
    model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', MultinomialNB())
    ])
    
    # Fit the model
    model.fit(sample_data['text'], sample_data['label'])
    
    # Test evaluation metrics
    metrics = evaluate_model(
        model, 
        sample_data['text'], 
        sample_data['label']
    )
    
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    
    # Check metric ranges
    for metric in metrics.values():
        assert 0 <= metric <= 1