import numpy as np
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

def detect_bias(y_test, y_pred, sensitive_features):
    if sensitive_features is None:
        return (None, None)
    
    # Convert inputs to appropriate types for Fairlearn
    import pandas as pd
    y_t = np.array(y_test).flatten()
    y_p = np.array(y_pred).flatten()
    
    # Convert sensitive features to pandas Series (Fairlearn prefers this)
    if isinstance(sensitive_features, pd.Series):
        sf = sensitive_features
    else:
        sf = pd.Series(np.array(sensitive_features).flatten())
    
    # Debug: Print shapes and types
    print(f"DEBUG: y_test shape: {y_t.shape}, type: {type(y_t)}")
    print(f"DEBUG: y_pred shape: {y_p.shape}, type: {type(y_p)}")
    print(f"DEBUG: sensitive_features shape: {sf.shape}, type: {type(sf)}")
    print(f"DEBUG: sensitive_features values: {sf.head() if len(sf) > 0 else 'Empty'}")
    
    # Ensure all arrays are 1D and have same length
    if len(y_t) != len(y_p) or len(y_t) != len(sf):
        raise ValueError(f"Length mismatch: y_test={len(y_t)}, y_pred={len(y_p)}, sensitive_features={len(sf)}")
    
    unique_vals = np.unique(np.concatenate((y_t, y_p)))
    
    if not set(unique_vals).issubset({0, 1}):
        top_class = pd.Series(y_t).mode()[0]
        y_t = (y_t == top_class).astype(int)
        y_p = (y_p == top_class).astype(int)
    else:
        y_t = y_t.astype(int)
        y_p = y_p.astype(int)
        
    # Standard Fairlearn metrics with comprehensive error handling
    try:
        print(f"DEBUG: Attempting Fairlearn with y_t shape: {y_t.shape}, sf shape: {sf.shape}")
        dpd = demographic_parity_difference(y_t, y_p, sensitive_features=sf)
        print(f"DEBUG: DPD calculation successful: {dpd}")
    except Exception as e:
        print(f"DEBUG: DPD failed with error: {e}")
        print(f"DEBUG: Error type: {type(e)}")
        # Try alternative approach - convert to list
        try:
            dpd = demographic_parity_difference(y_t.tolist(), y_p.tolist(), sensitive_features=sf.tolist())
            print(f"DEBUG: DPD successful with list conversion: {dpd}")
        except Exception as e2:
            print(f"DEBUG: DPD failed even with list conversion: {e2}")
            raise e2
    
    try:
        eod = equalized_odds_difference(y_t, y_p, sensitive_features=sf)
        print(f"DEBUG: EOD calculation successful: {eod}")
    except Exception as e:
        print(f"DEBUG: EOD failed with error: {e}")
        # Try alternative approach
        try:
            eod = equalized_odds_difference(y_t.tolist(), y_p.tolist(), sensitive_features=sf.tolist())
            print(f"DEBUG: EOD successful with list conversion: {eod}")
        except Exception as e2:
            print(f"DEBUG: EOD failed even with list conversion: {e2}")
            raise e2
    
    unique_groups = np.unique(sf)
    approval_rates = {}
    for group in unique_groups:
        mask = (sf == group)
        if np.sum(mask) > 0:
            approval_rates[str(group)] = float(np.mean(y_p[mask]))
        else:
            approval_rates[str(group)] = 0.0
            
    if len(approval_rates) > 1:
        rates = list(approval_rates.values())
        min_rate = min(rates)
        max_rate = max(rates)
        disparate_impact = min_rate / max_rate if max_rate > 0 else 0.0
    else:
        disparate_impact = 1.0
        
    metrics = {
        'Demographic Parity Difference': float(dpd),
        'Equal Opportunity Difference': float(eod),
        'Disparate Impact': float(disparate_impact),
        'Statistical Parity Ratio': float(disparate_impact) # Often used interchangeably
    }
    return (metrics, approval_rates)

def detect_intersectional_bias(y_test, y_pred, df_sensitive):
    """
    Analyzes bias across combinations of sensitive attributes.
    df_sensitive: pd.DataFrame with multiple sensitive columns
    """
    if df_sensitive is None or df_sensitive.empty:
        return (None, None)
    
    # Robustly create intersectional groups using vectorized string operations
    # This avoids 'apply' and 'join' which can hit TypeError in specific pandas/python versions
    try:
        # Convert all selected columns to string and handle NaNs
        df_str = df_sensitive.astype(str).replace(['nan', 'None', 'NaN'], 'Unknown')
        
        # Start with the first column
        intersectional_groups = df_str.iloc[:, 0]
        
        # Vectorized concatenation for all subsequent columns
        for i in range(1, df_str.shape[1]):
            intersectional_groups = intersectional_groups + " | " + df_str.iloc[:, i]
            
        return detect_bias(y_test, y_pred, intersectional_groups.values)
    except Exception as e:
        # Fallback for unexpected structural issues
        import pandas as pd
        fallback = df_sensitive.apply(lambda row: " | ".join([str(val) for val in row]), axis=1)
        return detect_bias(y_test, y_pred, fallback.values)

def classify_risk(disparate_impact):
    if disparate_impact < 0.8:
        return ('High Risk', '#DC2626')
    elif disparate_impact < 0.9:
        return ('Moderate Risk', '#F59E0B')
    else:
        return ('Fair', '#16A34A')
