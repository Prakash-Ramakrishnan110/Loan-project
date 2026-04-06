import numpy as np
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

def detect_bias(y_test, y_pred, sensitive_features):
    if sensitive_features is None:
        return None, None
        
    dpd = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_features)
    eod = equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive_features)
    
    unique_groups = np.unique(sensitive_features)
    approval_rates = {}
    
    for group in unique_groups:
        mask = (sensitive_features == group)
        if sum(mask) > 0:
            approval_rates[group] = np.mean(y_pred[mask])
        else:
            approval_rates[group] = 0.0
            
    if len(approval_rates) > 0:
        min_rate = min(approval_rates.values())
        max_rate = max(approval_rates.values())
        disparate_impact = min_rate / max_rate if max_rate > 0 else 0
    else:
        disparate_impact = 1.0
        
    metrics = {
        'Demographic Parity Difference': dpd,
        'Equal Opportunity Difference': eod,
        'Disparate Impact': disparate_impact
    }
    
    return metrics, approval_rates
