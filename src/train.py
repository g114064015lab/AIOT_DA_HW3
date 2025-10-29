"""Model training utilities with cross-validation and hyperparameter tuning."""
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .preprocess import build_vectorizer, preprocess_text


def create_pipeline(model_type: str = 'nb',
                   vectorizer_params: Optional[Dict[str, Any]] = None,
                   model_params: Optional[Dict[str, Any]] = None) -> Pipeline:
    """Create a pipeline with TF-IDF vectorizer and specified model."""
    if vectorizer_params is None:
        vectorizer_params = {}
    if model_params is None:
        model_params = {}
    
    vectorizer = build_vectorizer(**vectorizer_params)
    
    model_map = {
        'nb': MultinomialNB,
        'lr': LogisticRegression,
        'svm': LinearSVC,
        'rf': RandomForestClassifier
    }
    
    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}. Choose from: {list(model_map.keys())}")
    
    model = model_map[model_type](**model_params)
    return Pipeline([('tfidf', vectorizer), ('clf', model)])


def get_param_grid(model_type: str) -> Dict[str, Any]:
    """Get default parameter grid for GridSearchCV based on model type."""
    base_tfidf = {
        'tfidf__max_df': [0.5, 0.75, 0.9],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
    }
    
    model_grids = {
        'nb': {
            'clf__alpha': [0.1, 0.5, 1.0],
        },
        'lr': {
            'clf__C': [0.1, 1.0, 10.0],
            'clf__class_weight': [None, 'balanced'],
        },
        'svm': {
            'clf__C': [0.1, 1.0, 10.0],
            'clf__class_weight': [None, 'balanced'],
        },
        'rf': {
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [None, 10, 20],
            'clf__class_weight': [None, 'balanced'],
        }
    }
    
    if model_type not in model_grids:
        raise ValueError(f"No parameter grid defined for model type: {model_type}")
    
    # Combine base TF-IDF params with model-specific params
    param_grid = {}
    param_grid.update(base_tfidf)
    param_grid.update(model_grids[model_type])
    return param_grid


def train_and_evaluate(X_train: np.ndarray,
                      X_test: np.ndarray,
                      y_train: np.ndarray,
                      y_test: np.ndarray,
                      model_type: str = 'nb',
                      do_grid_search: bool = True,
                      n_cv_folds: int = 5,
                      random_state: int = 42) -> Tuple[Pipeline, Dict[str, Any]]:
    """Train a model with optional grid search and return metrics."""
    # Create base pipeline. Only pass random_state / solver params to models that accept them.
    model_params = None
    if model_type == 'lr':
        # Use class weighting to handle imbalance and ensure convergence
        model_params = {'class_weight': 'balanced', 'solver': 'liblinear', 'max_iter': 1000, 'random_state': random_state}
    elif model_type in ('svm', 'rf'):
        model_params = {'random_state': random_state}

    pipeline = create_pipeline(model_type, model_params=model_params)
    
    if do_grid_search:
        # Run grid search CV
        param_grid = get_param_grid(model_type)
        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=n_cv_folds,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=1
        )
        grid.fit(X_train, y_train)
        pipeline = grid.best_estimator_
        logging.info("Best parameters: %s", grid.best_params_)
    else:
        # Basic CV scores
        pipeline.fit(X_train, y_train)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=n_cv_folds, scoring='f1_macro')
        logging.info("CV F1 scores: %s", cv_scores)
        logging.info("CV F1 mean±std: %.4f ± %.4f", cv_scores.mean(), cv_scores.std())
    
    # Evaluate on test set
    y_pred = pipeline.predict(X_test)
    
    metrics = {
        'classification_report': classification_report(y_test, y_pred, digits=4),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    return pipeline, metrics


def save_pipeline(pipeline: Pipeline, path: Path) -> None:
    """Save trained pipeline to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    logging.info("Saved pipeline to %s", path)