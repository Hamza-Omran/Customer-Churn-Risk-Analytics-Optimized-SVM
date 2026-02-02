# Customer Churn Risk Analytics with Optimized Support Vector Machines

A comprehensive machine learning project combining classical optimization theory with practical deployment to predict customer churn in banking. This work demonstrates how custom-optimized models can achieve competitive performance while maintaining interpretability and computational efficiency.

---

## 1. Project Overview

### Problem Statement

Customer churn represents a significant challenge in the banking sector, where acquiring new customers costs 5-7 times more than retaining existing ones. This project tackles the problem of identifying at-risk customers before they leave, enabling proactive retention strategies.

### Why This Matters

**Business Relevance:**
- Early churn detection enables targeted retention campaigns
- Reduces customer acquisition costs
- Improves long-term customer lifetime value

**Academic Relevance:**
- Demonstrates practical application of convex optimization theory
- Compares theoretical implementations against industry-standard libraries
- Bridges the gap between optimization research and production ML systems

The project shows that classical machine learning models, when properly optimized and explained, can rival black-box approaches while remaining interpretable - critical in regulated industries like banking.

---

## 2. Dataset

**Source:** Bank customer transaction and demographic data (academic coursework dataset)

**Size:** 28,382 customer records → 24,832 after cleaning

**Features (21 total):**

- **Demographics:** age, gender, dependents, occupation, city
- **Account Information:** vintage (tenure), branch code, net worth category  
- **Financial Metrics:** current balance, previous balances, quarterly averages
- **Transaction Data:** monthly credits, debits, balances
- **Temporal:** last transaction date

**Engineered Features:**
- `tenure_ratio = vintage / age` - relationship duration relative to customer age
- `activity_score = transaction_sum / tenure` - transaction frequency metric

**Target Variable:** Binary churn (0 = stayed, 1 = churned)

**Class Distribution:** ~18.5% churn rate (imbalanced)

**Data Split:**
- Training: 70% (17,392 samples)
- Validation: 15% (3,715 samples)
- Test: 15% (3,725 samples)

**Preprocessing:**
- Missing value handling via removal (data sufficiency ensured)
- Outlier filtering (age constraints: 1-100 years)
- RobustScaler standardization (handles outliers better than StandardScaler)
- Label encoding for categorical variables

---

## 3. Methodology

### Data Preprocessing Pipeline

1. **Cleaning:** Removed records with missing values and invalid ages
2. **Feature Engineering:** Created tenure_ratio and activity_score
3. **Encoding:** LabelEncoder for categorical features (gender, occupation, city)
4. **Scaling:** RobustScaler for numerical features (robust to outliers)
5. **Splitting:** Stratified split to maintain class distribution

### Models Implemented

#### 3.1 Baseline: scikit-learn SVM

Standard implementation using:
- RBF (Radial Basis Function) kernel
- GridSearchCV hyperparameter tuning
- Parameters searched: C ∈ {1, 10, 100}, gamma ∈ {scale, auto}, class_weight ∈ {balanced, None}
- Best configuration: C=1, gamma=auto, class_weight=None

**Purpose:** Industry-standard baseline for comparison

#### 3.2 Custom: Gradient Descent SVM

From-scratch implementation featuring:
- Smoothed hinge loss for differentiability
- L2 regularization (λ = 0.01)
- Fixed learning rate (α = 0.001)
- 1000 iterations with full-batch updates

**Purpose:** Demonstrate understanding of smooth convex optimization

#### 3.3 Custom: Subgradient Descent SVM

Advanced implementation using:
- Standard (non-smooth) hinge loss
- Mini-batch updates (batch_size = 32)
- Dynamic learning rate: α_t = α / (1 + 0.01t)
- L2 regularization (λ = 0.0005)

**Purpose:** Show handling of non-differentiable loss functions

---

## 4. Optimization Contribution

This section explains the theoretical foundation and practical implementation of custom SVM optimizers.

### 4.1 Gradient Descent SVM (Smoothed Hinge Loss)

**Mathematical Formulation:**

The objective function minimizes:

```
L(w,b) = (1/n) Σ smoothed_hinge(y_i(w·x_i + b)) + λ||w||²
```

Where the smoothed hinge loss is defined as:

```
smoothed_hinge(t) = { 0,           if t ≥ 1
                     { (1-t)²/2,    if 0 < t < 1
                     { 1-t,         if t ≤ 0
```

**Why Smoothing Matters:**

The standard hinge loss is non-differentiable at t=1, making gradient-based optimization problematic. The smoothed version provides:
- Continuous differentiability everywhere
- Stable gradient computation
- Smoother convergence behavior

**Gradient Computation:**

```
∂L/∂w = { 2λw,                        if margin ≥ 1
        { 2λw - (1-margin)y_i·x_i,   if 0 < margin < 1
        { 2λw - y_i·x_i,              if margin ≤ 0
```

**Trade-off:** Smoother convergence but slower per-iteration updates compared to sklearn's optimized solvers.

### 4.2 Subgradient Descent SVM (Standard Hinge Loss)

**Mathematical Formulation:**

Minimizes the non-smooth objective:

```
L(w,b) = (1/n) Σ max(0, 1 - y_i(w·x_i + b)) + λ||w||²
```

**Subgradient Calculation:**

Since hinge loss is non-differentiable at margin=1, we use subgradients:

```
∂L/∂w ∈ { 2λw,              if margin > 1
        { [-y_i·x_i, 0],    if margin = 1  (set-valued)
        { 2λw - y_i·x_i,    if margin < 1
```

**Mini-batch Strategy:**

Instead of full-batch updates:
1. Shuffle training data each epoch
2. Process in batches of 32 samples
3. Compute subgradient average over batch
4. Apply learning rate decay: α_t = α/(1 + 0.01t)

**Why This Matters:**

- Handles non-smooth optimization (important theoretical skill)
- Mini-batching improves convergence speed
- Demonstrates practical optimization beyond textbook examples

### 4.3 Optimization Insights

**Convergence Comparison:**
- Gradient Descent: Smooth, stable convergence but computationally expensive
- Subgradient Descent: Faster convergence despite non-smoothness due to mini-batching
- sklearn SVM: Fastest due to highly optimized C++ backend (LibSVM/LibLinear)

**Theoretical Contribution:**
- Empirically validates that subgradient methods can be competitive
- Shows practical impact of smoothing techniques
- Demonstrates trade-offs between theoretical elegance and computational efficiency

---

## 5. Results & Comparison

### Performance Metrics

| Model | Train Accuracy | Test Accuracy | Training Time |
|-------|---------------|---------------|---------------|
| **sklearn SVM (RBF)** | 88.5% | **84.1%** | 641s |
| **Gradient Descent SVM** | 83.5% | 83.9% | 146s |
| **Subgradient Descent SVM** | 82.3% | 82.0% | 69s |

### Key Findings

1. **sklearn SVM achieves highest accuracy** thanks to:
   - Optimized kernel computations
   - Advanced solver (Sequential Minimal Optimization)
   - Extensive hyperparameter tuning

2. **Custom implementations are competitive:**
   - GD-SVM: 83.9% test accuracy (only 0.2% below sklearn)
   - SubGD-SVM: 82.0% test accuracy with **9.3x faster training**

3. **Speed-Accuracy Trade-off:**
   - Subgradient descent trains 9x faster than sklearn
   - Sacrifices only ~2% accuracy
   - Ideal for rapid prototyping or resource-constrained environments

4. **Feature Engineering Impact:**
   - tenure_ratio and activity_score improved all models
   - Shows domain knowledge can enhance even simple models

5. **Model Interpretability:**
   - Custom SVMs provide direct access to decision boundaries
   - Feature weights easily inspectable
   - Important for regulated banking applications

### Visualizations

Generated comparisons include:
- **Accuracy plots:** Train/validation/test performance across models
- **Training time comparison:** Bar charts showing computational efficiency
- **Convergence curves:** Loss over iterations for GD and SubGD
- **Confusion matrices:** Error analysis for all three models
- **ROC curves:** Probability calibration assessment

---

## 6. Deployment

A production-ready Streamlit web application provides:

### Features

**1. Home Dashboard**
- Project overview and methodology
- Performance metrics summary
- Academic context

**2. Single Prediction**
- Interactive form for customer data entry
- Real-time risk score calculation (0-100%)
- Risk level classification (Low/Medium/High)
- Prediction explanation

**3. Batch Prediction**
- CSV file upload support
- Bulk processing of customer records
- Downloadable results with risk scores

**4. Model Comparison**
- Side-by-side performance visualization
- Training time analysis
- Interactive charts

### Running Locally

**1. Install Dependencies**
```bash
pip install -r requirements.txt
```

**2. Train Models** (optional - models already included)
```bash
python3 train_models.py
```
Takes ~15 minutes. Trains all three SVMs and generates visualizations.

**3. Launch Web App**
```bash
streamlit run deployment/app.py
```

Opens at `http://localhost:8501`

### Technology Stack

- **Backend:** Python 3.x, scikit-learn, NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** Streamlit
- **Model Persistence:** joblib

---

## 7. Screenshots

### Dashboard Home
*Professional overview with metrics and academic context*

<img width="1914" height="970" alt="image" src="https://github.com/user-attachments/assets/b836b5c9-0f25-4cf7-b762-1564de86c39d" />


### Model Performance Comparison
*Side-by-side accuracy and training time analysis*

<img width="4167" height="1470" alt="image" src="https://github.com/user-attachments/assets/bcf96213-bc32-41e4-b1e5-68f760a7148d" />


### Convergence Analysis
*Loss curves showing optimization behavior*

<img width="2967" height="1768" alt="image" src="https://github.com/user-attachments/assets/4d0bf24e-29be-4c99-8da6-270e02552432" />

### Confusion Matrices
*Error analysis across all three models*

<img width="4426" height="1168" alt="image" src="https://github.com/user-attachments/assets/df010ba4-cc65-4706-89c8-108686447446" />

### ROC Curve
*Probability calibration for sklearn SVM*

<img width="2367" height="1768" alt="image" src="https://github.com/user-attachments/assets/8f9ce7e7-ac69-47f6-a48b-abab60cb77d9" />

---

## 8. Academic Context

### Relevance for MSc Programs

**Suitable for:**
- MSc in Artificial Intelligence
- MSc in Data Science
- MSc in Machine Learning
- Erasmus Mundus AI programs

**Demonstrates competencies in:**
- **Machine Learning:** Classification, model selection, evaluation
- **Optimization Theory:** Convex optimization, gradient methods, subgradient methods
- **Software Engineering:** Modular code, deployment, version control
- **Statistical Analysis:** Feature engineering, model comparison, hypothesis testing
- **Communication:** Technical writing, visualization, presentation

### Extension to Thesis Work

This project provides foundation for:

1. **Advanced Optimization Research:**
   - Adaptive learning rate strategies
   - Stochastic variance reduction methods
   - Proximal gradient methods for structured sparsity

2. **Model Interpretability:**
   - SHAP value analysis
   - Counterfactual explanations
   - Fairness considerations in churn prediction

3. **Deployment at Scale:**
   - Production pipeline design
   - A/B testing framework
   - Real-time prediction infrastructure

4. **Domain Applications:**
   - Healthcare patient retention
   - Telecom churn prediction
   - SaaS subscription modeling

---

## 9. Related Work

This project builds on and integrates concepts from:

### Previous Projects

**[Bank Churn Prediction ML Pipeline](https://github.com/yourusername/Bank-Churn-Prediction-ML-Pipeline)**
- Original churn analysis with feature selection
- Established baseline SVM performance
- Provided dataset and initial EDA

**[SVM Gradient & Subgradient Optimization](https://github.com/yourusername/SVM-Gradient-Subgradient-Banknotes)**
- Theoretical foundation for custom SVM implementations
- Banknote authentication problem
- Initial optimization experiments

**[Titanic Survival Analysis](https://github.com/yourusername/Titanic-Survival-Analysis-and-Prediction)**
- Demonstrates ML workflow generalization
- Feature engineering techniques
- Model comparison methodology

**[Diamond Price Prediction](https://github.com/yourusername/Diamond-Price-Prediction-Regression-Model)**
- Statistical analysis expertise
- Regression modeling
- Data visualization skills

### Key References

1. Vapnik, V. N. (1995). *The Nature of Statistical Learning Theory*. Springer.
2. Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.
3. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
4. Shalev-Shwartz, S., & Ben-David, S. (2014). *Understanding Machine Learning*. Cambridge University Press.

---

## 10. Project Structure

```
Customer-Churn-Risk-Analytics-Optimized-SVM/
├── data/
│   └── raw/                    # Original dataset
├── deployment/
│   ├── app.py                  # Streamlit web application
│   └── models/                 # Trained models (sklearn, GD, SubGD, scaler)
├── docs/
│   └── ARCHITECTURE.md         # System architecture documentation
├── results/
│   ├── comparisons/            # Performance metrics CSV
│   └── figures/                # Visualization images
├── src/
│   ├── models/                 # Custom SVM implementations
│   ├── preprocessing/          # Data pipeline
│   └── explainability/         # Feature importance utilities
├── train_models.py             # Main training script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 11. Installation & Usage

**Clone Repository**
```bash
git clone <repository-url>
cd Customer-Churn-Risk-Analytics-Optimized-SVM
```

**Install Requirements**
```bash
pip install -r requirements.txt
```

**Train Models** (optional)
```bash
python3 train_models.py
```

**Run Web Application**
```bash
streamlit run deployment/app.py
```

---

## 12. Team & Acknowledgments

**Course:** Data Computation Analysis (2025)

**Team Members:**
- Hamza Hussain Omran - 22011501
- Ahmad Khalifa Abdulraouf Mohamad - 22010018
- Moaz Moustafa Abd-Elhamid - 22010263
- Abdurhman Hesham Ragab - 22010136
- Kareem Mohamed Samy Aboshady - 22010378

**Contributions:**
This implementation integrates work from the Bank Churn prediction project with custom optimization algorithms developed for convex optimization coursework.

---

## 13. License

Academic use only. This project is part of MSc coursework and maintained for educational purposes.

---

## 14. Future Work

Potential extensions include:

- **Optimization:** Implement accelerated gradient methods (Nesterov, ADAM)
- **Models:** Test kernel approximations (Nyström, Random Fourier Features)
- **Explainability:** Add SHAP values and counterfactual generation
- **Deployment:** Containerization with Docker, REST API development
- **Scalability:** Implement online learning for streaming data

---

**For questions or collaboration:**  
Contact: [your-email]  
GitHub: [your-github]
