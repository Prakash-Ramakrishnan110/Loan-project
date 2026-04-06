def generate_report(metrics_before, bias_before, metrics_after=None, bias_after=None):
    report = "# Automated Fairness Audit & Compliance Report\n\n"
    
    report += "## 1. Initial Model Performance\n"
    for k, v in metrics_before.items():
        report += f"- **{k}**: {v:.4f}\n"
        
    report += "\n## 2. Initial Bias Metrics\n"
    for k, v in bias_before.items():
        if isinstance(v, float):
            report += f"- **{k}**: {v:.4f}\n"
        else:
            report += f"- **{k}**: {v}\n"
            
    if metrics_after and bias_after:
        report += "\n## 3. Post-Mitigation Performance\n"
        for k, v in metrics_after.items():
            report += f"- **{k}**: {v:.4f}\n"
            
        report += "\n## 4. Post-Mitigation Bias Metrics\n"
        for k, v in bias_after.items():
            if isinstance(v, float):
                report += f"- **{k}**: {v:.4f}\n"
            else:
                report += f"- **{k}**: {v}\n"
                
    report += "\n\n---\n*System Generate Report - Validator 1.0*"
    return report
