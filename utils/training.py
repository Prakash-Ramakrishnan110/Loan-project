from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

def train_model(X, y, sensitive_features=None, model_type='Logistic Regression'):
    # If sensitive features provided, stratify or just standard split
    # We will do a standard split but keep track of sensitive features if present
    
    if sensitive_features is not None:
        X_train, X_test, y_train, y_test, sf_train, sf_test = train_test_split(
            X, y, sensitive_features, test_size=0.2, random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        sf_train, sf_test = None, None
        
    if model_type == 'Random Forest':
        model = RandomForestClassifier(random_state=42, n_estimators=100)
    else:
        model = LogisticRegression(random_state=42, max_iter=1000)
        
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred)
    }
    
    return model, metrics, X_test, y_test, y_pred, X_train, y_train, sf_train, sf_test
