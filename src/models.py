"""
models.py - lightweight ML model helpers.
Includes:
- unsupervised IsolationForest detector
- supervised RandomForest classifier (if labels provided)
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score
from joblib import dump, load
from preprocess import load_data
from pathlib import Path

CLASS_LABELS = {
    0: 'abnormal',
    1: 'normal'
}

def train_supervised(X: pd.DataFrame, y: pd.Series, models: str = "rf"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique()>1 else None)
    
    # Random Forest Classifier
    if models == "rf":
        clf = RandomForestClassifier(n_estimators=200, random_state=42)

    # Logistic Regression Classifier
    elif models == "lr":
        clf = LogisticRegression(max_iter=10000, random_state=0)
    
    # LightBGM Classifier
    elif models == "lgbm":
        clf = LGBMClassifier(objective='binary', random_state=42)

    else:
        raise ValueError("Unsupported model type. Choose from 'rf', 'lr', or 'lgbm'.")

    clf.fit(X_train, y_train)
    score = clf.score(X_train, y_train)
    print("Acc. Train:", score)
    score = clf.score(X_test, y_test)
    print("Acc. Test:", score)
    
    # X_pred = clf.predict(X_test)
    X_prob = clf.predict_proba(X_test)

    y_prob = np.array(y_test.copy())
    y_prob[y_prob=='abnormal'] = 0
    y_prob[y_prob=='normal'] = 1

    # print(X_prob[y_prob])
    score = mean_squared_error(X_prob[:,1], y_prob)
    print("MSE Test:", np.round(score, 3))
    
    X_pred = np.argmax(X_prob, axis=-1)
    score = precision_score(X_pred, y_prob.astype(int))
    print("precision Test:", np.round(score, 3))
    score = recall_score(X_pred, y_prob.astype(int))
    print("recall Test:", np.round(score, 3))
    score = f1_score(X_pred, y_prob.astype(int))
    print("f1_score Test:", np.round(score, 3))
    return clf

def score_supervised(model, X: pd.DataFrame):
    score = model.predict_proba(X)
    label = model.predict(X)
    # pred = CLASS_LABELS[np.argmax(X_prob)]
    return score[0][0], label[0]  # return prob of "abnormal" class and label

def save_model(model, path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    dump(model, path)

def load_model(path: str):
    return load(path)