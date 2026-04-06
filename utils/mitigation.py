from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

def mitigate_bias(X_train, y_train, sensitive_features_train, model_type='Logistic Regression', method='Exponentiated Gradient'):
    if model_type == 'Random Forest':
        base_model = RandomForestClassifier(random_state=42, n_estimators=50)
    else:
        base_model = LogisticRegression(random_state=42, max_iter=1000)
        
    if method == 'Exponentiated Gradient (Fairlearn)':
        mitigator = ExponentiatedGradient(base_model, constraints=DemographicParity())
        # We need to make sure y_train is passed correctly
        mitigator.fit(X_train, y_train, sensitive_features=sensitive_features_train)
        return mitigator
    elif method == 'Reweighing (AIF360)':
        try:
            from aif360.algorithms.preprocessing import Reweighing
            from aif360.datasets import BinaryLabelDataset
            
            df = pd.DataFrame(X_train.copy())
            df_cols = df.columns
            df['target'] = np.array(y_train)
            df['sensitive'] = np.array(sensitive_features_train)
            
            mode_val = pd.Series(sensitive_features_train).mode()[0]
            privileged_groups = [{'sensitive': mode_val}]
            unprivileged_classes = [v for v in np.unique(sensitive_features_train) if v != mode_val]
            unprivileged_groups = [{'sensitive': v} for v in unprivileged_classes]
            
            dataset = BinaryLabelDataset(df=df, label_names=['target'], protected_attribute_names=['sensitive'])
            
            RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
            dataset_transf = RW.fit_transform(dataset)
            
            weights = dataset_transf.instance_weights
            base_model.fit(X_train, y_train, sample_weight=weights)
            
            class ReweighedModelWrapper:
                def __init__(self, model):
                    self.model = model
                def predict(self, X):
                    return self.model.predict(X)
                    
            return ReweighedModelWrapper(base_model)
        except Exception as e:
            # Fallback if aif360 fails
            base_model.fit(X_train, y_train)
            return base_model
    else:
        base_model.fit(X_train, y_train)
        return base_model
