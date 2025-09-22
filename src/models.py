"""
models.py - lightweight ML model helpers.
Includes:
- unsupervised IsolationForest detector
- supervised RandomForest classifier (if labels provided)
"""
import numpy as np
import pandas as pd
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

def train_isolation_forest(X: pd.DataFrame, random_state: int = 42, **kwargs):
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=random_state, **kwargs)
    model.fit(X)
    return model

def score_isolation_forest(model, X: pd.DataFrame):
    # sklearn's IsolationForest.score_samples returns opposite sign: lower -> more abnormal.
    # we convert to anomaly score in [0,1] where 1 == most anomalous
    raw = -model.score_samples(X)  # higher values -> more anomalous
    # normalize to 0..1
    raw_min, raw_max = raw.min(), raw.max()
    if raw_max - raw_min <= 0:
        return np.zeros_like(raw)
    return (raw - raw_min) / (raw_max - raw_min)

def train_supervised_rf(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique()>1 else None)
    
    # Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=200, random_state=42)

    # Logistic Regression Classifier
    # clf = LogisticRegression(max_iter=10000, random_state=0)

    clf.fit(X_train, y_train)
    score = clf.score(X_train, y_train)
    print("Acc. Train:", score)
    score = clf.score(X_test, y_test)
    print("Acc. Test:", score)
    
    # X_pred = clf.predict(X_test)
    X_prob = clf.predict_proba(X_test)

    # print(X_prob[:,1])
    # import sys
    # sys.exit()
    
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

def score_supervised_rf(model, X: pd.DataFrame):
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

# TEST
df = load_data("C:/temp/Pegatron/test/smart-factory-agent/data/sensors_processed.csv")
label = df['label']
df = df.drop(columns=['label', 'timestamp'])
# model = train_supervised_rf(df, label)
# save_model(model, "model/randomforest.pth")
# print(df.tail(1)['temp'])