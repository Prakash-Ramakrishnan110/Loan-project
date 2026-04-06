import shap
import matplotlib.pyplot as plt
import numpy as np

def generate_shap_plots(model, X_train, X_test, model_type='Logistic Regression'):
    if hasattr(model, 'predictors_'):
        shap_model = model.predictors_[0]
    elif hasattr(model, 'model'):
        shap_model = model.model
    else:
        shap_model = model
        
    fig_summary, ax = plt.subplots(figsize=(8, 5))
    
    # Take a small sample to avoid long compute times
    X_sample = X_test[:100]
    
    try:
        if model_type == 'Random Forest':
            explainer = shap.TreeExplainer(shap_model)
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap.summary_plot(shap_values[1], X_sample, show=False)
            else:
                shap.summary_plot(shap_values, X_sample, show=False)
        else:
            explainer = shap.LinearExplainer(shap_model, X_train)
            shap_values = explainer.shap_values(X_sample)
            shap.summary_plot(shap_values, X_sample, show=False)
    except Exception as e:
        ax.text(0.5, 0.5, 'SHAP generation failed or not supported for this model type', 
                ha='center', va='center', transform=ax.transAxes)
                
    plt.tight_layout()
    return fig_summary
