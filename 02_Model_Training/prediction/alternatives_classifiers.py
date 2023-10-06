
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from utils.evaluation_classifier import evaluate_classifier, calculate_map

def train_alternative_classifier(X_train, X_dev, X_test, y_train, y_dev, y_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_dev = scaler.transform(X_dev)
    X_test = scaler.transform(X_test)

    classifiers = {
        'SVM': SVC(probability=True),
        'Random Forest': RandomForestClassifier(),
        'Linear Classifier': LogisticRegression(max_iter=10000),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    for name, classifier in classifiers.items():
        print(f'Training {name}...')
        classifier.fit(X_train, y_train)
        print(f'Evaluating {name} on dev set...')
        evaluate_classifier(classifier, X_dev, y_dev, name)

    print('Final evaluation on test set...')
    for name, classifier in classifiers.items():
        evaluate_classifier(classifier, X_test, y_test, name)