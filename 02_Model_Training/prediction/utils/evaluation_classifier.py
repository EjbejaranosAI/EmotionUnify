from sklearn.metrics import average_precision_score, f1_score, accuracy_score

def calculate_map(y_true, y_probs):
    return average_precision_score(y_true, y_probs, average="macro")
def evaluate_classifier(classifier, X, y, name):
    preds = classifier.predict(X)
    if hasattr(classifier, 'predict_proba'):
        probs = classifier.predict_proba(X)
        map_score = calculate_map(y, probs)
    else:
        map_score = 'N/A'
    accuracy = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average='weighted')
    print(f'{name} - Accuracy: {accuracy * 100:.2f}%, F1 Score: {f1 * 100:.2f}%, MaP: {map_score if map_score == "N/A" else f"{map_score * 100:.2f}%"}')