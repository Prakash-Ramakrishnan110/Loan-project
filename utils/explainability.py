import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _unwrap_model(model):
    """Extract the underlying sklearn estimator from wrappers."""
    if hasattr(model, "predictors_"):
        return model.predictors_[0]
    if hasattr(model, "model"):
        return model.model
    return model


def compute_shap_values(model, X_train, X_test, model_type="Logistic Regression"):
    """
    Compute SHAP values for the given model and return (shap_values, explainer).
    Uses a subsample of X_test to keep computation fast.
    """
    shap_model = _unwrap_model(model)
    X_sample = X_test[:100] if len(X_test) > 100 else X_test

    if model_type == "Random Forest":
        explainer = shap.TreeExplainer(shap_model)
    else:
        explainer = shap.LinearExplainer(shap_model, X_train)

    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return shap_values, X_sample


def get_feature_importance(shap_values, feature_names):
    """
    Compute mean |SHAP| per feature and return sorted DataFrame.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": mean_abs,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    return importance_df


def generate_shap_summary_plot(model, X_train, X_test, model_type="Logistic Regression"):
    """Generate a matplotlib SHAP summary plot figure."""
    shap_model = _unwrap_model(model)
    X_sample = X_test[:100] if len(X_test) > 100 else X_test

    fig, ax = plt.subplots(figsize=(10, 6))

    try:
        if model_type == "Random Forest":
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
    except Exception:
        ax.text(
            0.5, 0.5,
            "SHAP computation failed for this model configuration",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=12, color="#64748B",
        )

    plt.tight_layout()
    return plt.gcf()
