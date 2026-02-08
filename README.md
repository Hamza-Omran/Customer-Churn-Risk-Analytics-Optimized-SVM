# Customer Churn Risk Analytics with Optimized Support Vector Machines

End-to-end machine learning system for predicting bank customer churn, combining convex optimization theory with practical deployment and interpretability.

**Context:** Machine Learning / Optimization course project (2024)

---

## Problem Overview

Customer churn is a critical challenge in banking, where customer acquisition costs significantly exceed retention costs.  
This project focuses on identifying high-risk customers early to enable proactive retention strategies.

**Key focus:**  
Building **interpretable, efficient models** using classical optimization rather than black-box approaches.

---

## Core Implementation

### Models Implemented

- **Baseline:** scikit-learn SVM (RBF kernel, GridSearchCV)
- **Custom SVM (Gradient Descent):**
  - Smoothed hinge loss
  - L2 regularization
  - Full-batch optimization
- **Custom SVM (Subgradient Descent):**
  - Standard hinge loss
  - Mini-batch training
  - Learning-rate decay

---

## Technical Highlights

- From-scratch implementation of **gradient and subgradient SVM optimization**
- Handling **non-differentiable loss functions**
- Mini-batch optimization for faster convergence
- Robust preprocessing pipeline (outliers, imbalance handling)
- Feature engineering informed by domain knowledge
- Model interpretability via explicit decision boundaries
- End-to-end deployment using Streamlit

---

## Key Insights

- Subgradient descent achieves **~9× faster training** with only ~2% accuracy loss
- Smoothed hinge loss stabilizes convergence but increases computation cost
- Feature engineering improves all models more than hyperparameter tuning alone
- Classical models remain competitive when optimization is done carefully
- Interpretability is preserved — critical for regulated domains like banking

---

## Results Summary

| Model | Test Accuracy | Training Time |
|------|---------------|---------------|
| sklearn SVM (RBF) | **84.1%** | 641s |
| Gradient Descent SVM | 83.9% | 146s |
| Subgradient Descent SVM | 82.0% | **69s** |

---

## Deployment

A production-ready Streamlit application provides:

- Single-customer churn risk prediction
- Batch prediction via CSV upload
- Model comparison dashboards
- Visual performance analysis

**Run locally:**
```bash
pip install -r requirements.txt
streamlit run deployment/app.py

---

## Interface Preview

-- Selected screenshots from the deployed application, illustrating system functionality and analytical capabilities rather than UI design.

-- Dashboard Overview
<img src="https://github.com/user-attachments/assets/b836b5c9-0f25-4cf7-b762-1564de86c39d" width="800"/>
-- Model Performance Comparison
<img src="https://github.com/user-attachments/assets/bcf96213-bc32-41e4-b1e5-68f760a7148d" width="800"/>
-- Optimization Convergence Analysis
<img src="https://github.com/user-attachments/assets/4d0bf24e-29be-4c99-8da6-270e02552432" width="800"/>
