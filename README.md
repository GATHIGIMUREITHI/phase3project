Telecom Customer Churn Prediction


A machine learning project to predict customer churn for SyriaTel Telecommunications Company, enabling proactive retention strategies and reducing customer attrition.


---

Table of Contents

- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [Model Performance](#model-performance)
- [Business Recommendations](#business-recommendations)
- [Usage](#usage)
- [Future Work](#future-work)
- [Contributors](#contributors)
- [License](#license)

---

 Project Overview

Customer churn is a critical challenge in the telecommunications industry, where acquiring new customers costs **5-25 times more** than retaining existing ones. This project builds a predictive model to:

-  Identify customers at high risk of churning before they leave
-  Understand key factors contributing to customer churn
-  Develop targeted retention strategies to reduce churn rate
-  Provide actionable insights for business stakeholders

Stakeholder: Executive Leadership Team at SyriaTel Telecommunications Company

---

Business Problem

SyriaTel is experiencing customer attrition and needs a data-driven approach to:

1. **Predict** which customers are likely to churn
2. **Identify** the strongest predictors of churn behavior
3. **Segment** customers by risk level for targeted interventions
4. **Optimize** retention strategies to maximize ROI

Success Criteria

- **Primary Metric**: Achieve **recall ≥ 70%** (catch most customers who will churn)
- **Rationale**: Missing a churning customer is more costly than false alarms
- **Business Impact**: Enable proactive retention campaigns with positive ROI

---

Dataset

**Source**: SyriaTel Customer Churn Dataset  
**Size**: 3,333 customers × 21 features  
**Target Variable**: `churn` (True/False)

Key Features

| Category | Features |
|----------|----------|
| **Demographics** | State, area code, account length |
| **Service Plans** | International plan, voice mail plan |
| **Usage Patterns** | Day/evening/night/international minutes, calls, charges |
| **Customer Service** | Number of customer service calls |

Class Distribution

- **Not Churned**: 2,850 customers (85.5%)
- **Churned**: 483 customers (14.5%)

---
Installation

 Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

 Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/GATHIGIMUREITHI/telecom-churn-prediction.git
   cd telecom-churn-prediction
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

 Required Libraries

```txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
jupyter>=1.0.0
```

---

Project Structure

```
telecom-churn-prediction/
│
├── data/
│   └── SyriaTelCustomerChurn.csv          # Raw dataset
│
├── notebooks/
│   └── churn_prediction_analysis.ipynb    # Main analysis notebook
│
├── images/                                 # Generated visualizations
│   ├── confusion_matrix_*.png
│   ├── roc_curve_*.png
│   ├── feature_importance_*.png
│   └── model_comparison.png
│
├── models/                                 # Saved model files (optional)
│
├── README.md                               # Project documentation
├── requirements.txt                        # Python dependencies
└── LICENSE                                 # License file
```

---

Methodology

### 1. **Business & Data Understanding**
   - Defined business problem and success criteria
   - Explored dataset structure and features
   - Analyzed target variable distribution

### 2. **Exploratory Data Analysis (EDA)**
   - Investigated churn patterns across features
   - Identified key correlations with churn
   - Visualized customer segments and behaviors

### 3. **Data Preparation**
   - **Feature Engineering**:
     - Created aggregate usage features (total minutes, calls, charges)
     - Calculated usage ratios (international usage ratio)
     - Generated risk flags (high customer service calls)
     - Derived revenue metrics (revenue per day, avg call duration)
   - **Encoding**: Binary encoding for yes/no features, one-hot encoding for categorical variables
   - **Scaling**: Standardized numerical features using StandardScaler
   - **Train-Test Split**: 80/20 split with stratification

### 4. **Modeling**

Three classification models were trained and evaluated:

| Model | Description | Key Strength |
|-------|-------------|--------------|
| **Logistic Regression** | Baseline linear model | High interpretability |
| **Decision Tree** | Non-linear rule-based classifier | Captures complex patterns |
| **Random Forest** | Ensemble of decision trees | Robust and accurate |

### 5. **Evaluation**

Models were evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: Accuracy of churn predictions
- **Recall**: Ability to catch churners (priority metric)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Discrimination capability

---

Key Findings

Top Churn Predictors

1. **Customer Service Calls** 
   - Customers with **4+ calls** have significantly higher churn rates
   - Indicates unresolved issues or dissatisfaction

2. **International Plan** 
   - Customers with international plans show distinct churn patterns
   - Suggests pricing or value perception issues

3. **Total Charges** 
   - High daily charges correlate with churn
   - Potential "bill shock" or competitor pricing advantages

4. **Voice Mail Plan** 
   - Customers with voice mail plans have **lower churn rates**
   - Value-added services increase retention

5. **Usage Patterns** 
   - International usage ratio is a strong predictor
   - Extreme usage (very high or very low) indicates risk

Customer Insights

- **High-Risk Segment**: International plan holders with high service calls
- **Low-Risk Segment**: Long-tenure customers with voice mail plans
- **Churn Trigger**: 4+ customer service calls is a critical threshold

---

Model Performance

Model Comparison

| Model | Test Accuracy | Test Precision | **Test Recall** | Test F1-Score | Test ROC-AUC |
|-------|---------------|----------------|-----------------|---------------|--------------|
| Logistic Regression | 85.2% | 47.9% | **23.7%** | 31.7% | 84.1% |
| **Decision Tree**  | **96.9%** | **100.0%** | **78.4%** | **87.9%** | 87.3% |
| Random Forest | 95.1% | 97.1% | 68.0% | 80.0% | 90.8% |

Final Model Selection: **Decision Tree Classifier**

**Rationale**:
-  Highest recall (78.4%) - catches most churners
-  Perfect precision (100%) - no false positives
-  High interpretability - easy to explain to stakeholders
-  Strong overall performance (96.9% accuracy)

Confusion Matrix (Decision Tree)

```
                Predicted
              Not Churn  Churn
Actual Not      570       0
       Churn     21      76
```

**Interpretation**:
- **True Negatives**: 570 (correctly identified loyal customers)
- **False Positives**: 0 (no false alarms)
- **False Negatives**: 21 (missed churners - 21.6% miss rate)
- **True Positives**: 76 (correctly caught churners - 78.4% catch rate)

---

Business Recommendations

Immediate Actions (High-Risk Customers)

#### 1. Customer Service Quality Improvement
- **Finding**: Customers with 4+ service calls are at extreme risk
- **Action**: 
  - Implement first-call resolution training
  - Create dedicated retention team for high-volume callers
  - Root cause analysis for recurring issues
- **Expected Impact**: 30-40% reduction in high-call-volume churn

#### 2. International Plan Optimization
- **Finding**: International plan holders have higher churn rates
- **Action**:
  - Review pricing and value proposition
  - Offer bundled services or promotional rates
  - Proactive outreach with usage optimization tips
- **Expected Impact**: 20% reduction in international plan churn

Medium-Term Strategies (At-Risk Segments)

#### 3. Usage-Based Interventions
- Monitor customers with high charges (bill shock prevention)
- Implement usage alerts and plan optimization recommendations
- Offer customized plans based on individual usage patterns

#### 4. Value-Added Services Promotion
- Promote voice mail plans to customers without them
- Bundle services to increase switching costs
- Create loyalty programs based on tenure

 Predictive Retention Program

#### 5. Risk-Based Scoring System
- **High Risk** (>70% churn probability): Personal call + special offers
- **Medium Risk** (40-70%): Automated email campaigns + value reminders
- **Low Risk** (<40%): Standard satisfaction surveys

#### 6. Financial Analysis Example

With 1,000 predicted churners:
- **Model Recall**: 78.4% → Identify 784 actual churners
- **Retention Campaign Success**: 30% → Retain 235 customers
- **Customer Lifetime Value**: $1,200 per customer
- **Retention Cost**: $100 per offer

**ROI Calculation**:
```
Value Saved: 235 customers × $1,200 = $282,000
Campaign Cost: 1,000 offers × $100 = $100,000
Net Benefit: $182,000
ROI: 182%
```

Continuous Improvement

- Deploy model for monthly customer scoring
- A/B test retention strategies
- Monitor model performance and retrain quarterly
- Collect feedback from churned customers to improve model

---

 Usage

### Running the Analysis

1. Open the main notebook:
   ```bash
   jupyter notebook notebooks/churn_prediction_analysis.ipynb
   ```

2. Run all cells sequentially to reproduce the analysis

3. Generated visualizations will be saved to the `images/` directory

### Using the Model for Predictions

```python
import pandas as pd
import pickle

# Load the trained model (if saved)
# with open('models/decision_tree_model.pkl', 'rb') as f:
#     model = pickle.load(f)

# Load new customer data
new_customers = pd.read_csv('new_customer_data.csv')

# Preprocess (apply same transformations as training)
# ... feature engineering, encoding, scaling ...

# Make predictions
churn_predictions = model.predict(new_customers_processed)
churn_probabilities = model.predict_proba(new_customers_processed)[:, 1]

# Identify high-risk customers
high_risk = new_customers[churn_probabilities > 0.7]
print(f"High-risk customers: {len(high_risk)}")
```

---

Future Work

### Model Enhancements
- [ ] Implement advanced ensemble methods (XGBoost, LightGBM)
- [ ] Hyperparameter tuning with Bayesian optimization
- [ ] Handle class imbalance with SMOTE or cost-sensitive learning
- [ ] Time-series analysis of churn patterns

### Feature Engineering
- [ ] Customer lifetime value calculation
- [ ] Behavioral trend features (increasing/decreasing usage)
- [ ] Geographic and demographic enrichment
- [ ] Competitor pricing and market data integration

### Deployment
- [ ] Create REST API for real-time predictions
- [ ] Build interactive dashboard for stakeholders
- [ ] Automate monthly scoring pipeline
- [ ] Integrate with CRM system for automated alerts

### Business Impact
- [ ] A/B test retention campaigns
- [ ] Track ROI of interventions
- [ ] Conduct customer exit interviews
- [ ] Benchmark against industry standards

---

Contributors

**Jeffrey Gathigi** - *Data Scientist*  
- GitHub: [@GATHIGIMUREITHI](https://github.com/GATHIGIMUREITHI)
- Instructor: Brian Chacha



---


---

---


---


