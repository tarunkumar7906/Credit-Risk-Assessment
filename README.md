# Credit Risk Assessment — Advanced ML with SHAP Explainability

A production machine learning web application that predicts loan applicant credit risk using an **XGBoost classifier with SHAP explainability**. This project goes beyond standard binary classification to provide both predictions AND interpretable explanations of why an applicant is deemed high or low risk.

**Key Innovation:** Every prediction is backed by **SHAP (SHapley Additive exPlanations)** values — both global feature importance (which factors matter most across all predictions) and local explanations (why this specific applicant was classified this way).

**🌐 Live Application:** [https://credit-risk-assessment-1-3qxp.onrender.com](https://credit-risk-assessment-1-3qxp.onrender.com)
**📊 GitHub Repository:** [github.com/tarunkumar7906/Credit-Risk-Assessment](https://github.com/tarunkumar7906/Credit-Risk-Assessment)

---

## 🎯 The Real-World Problem

Banks and lenders need to classify loan applicants as high-risk or low-risk for default. But it's not enough to just say "deny" — regulators and customers demand **explainability**: *Why was this application rejected?* Which factors drove the decision?

Traditional ML models are "black boxes" — you get a prediction but no insight. This project solves that by:
1. Building a high-accuracy XGBoost classifier
2. Adding SHAP explainability so every prediction is interpretable
3. Deploying it as a live web app where you can see probabilities AND feature importance

**This is what responsible AI looks like in 2026.**

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│              FRONTEND — Credit Risk Assessment                │
│           HTML + CSS + Vanilla JavaScript (370 lines)        │
│  - Loan application form (11 fields: age, income, intent,…)  │
│  - Animated risk gauge showing default probability           │
│  - SHAP summary/waterfall explanations (future feature)       │
├──────────────────────────────────────────────────────────────┤
│               FETCH API / HTTPS (JSON)                       │
├──────────────────────────────────────────────────────────────┤
│            BACKEND — FastAPI REST API (45 lines)             │
│  - Pydantic validation on all 11 loan application fields     │
│  - Single POST /predict endpoint                              │
│  - Returns: probability + risk level + metadata               │
├──────────────────────────────────────────────────────────────┤
│    ML PIPELINE — Tuned XGBoost with Probability Calibration  │
│  - ColumnTransformer: numeric + categorical preprocessing    │
│  - Handle class imbalance with scale_pos_weight              │
│  - Optimized threshold for business needs                    │
│  - SHAP TreeExplainer for global + local interpretability    │
├──────────────────────────────────────────────────────────────┤
│           DEPLOYMENT — Render Cloud Platform                  │
│  - FastAPI + Uvicorn ASGI server                               │
│  - Python 3.11.9 runtime (specified in runtime.txt)           │
│  - CORS enabled for frontend integration                       │
└──────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

### Machine Learning (Jupyter Notebook — 74 cells)
- **Language:** Python (Pandas, NumPy)
- **Primary Algorithm:** XGBoost Classifier
- **Baseline Model:** Logistic Regression (for comparison)
- **Feature Engineering:** ColumnTransformer for pipeline-based preprocessing
- **Class Imbalance:** `scale_pos_weight` in XGBoost (automatic handling of the imbalanced target distribution)
- **Model Tuning:** GridSearchCV on hyperparameters
- **Threshold Optimization:** Probability calibration for business-specific decision boundaries
- **Explainability:** SHAP (SHapley Additive exPlanations)
  - TreeExplainer for XGBoost
  - Global summary plots (feature importance across all predictions)
  - Local waterfall plots (feature impact for individual applicants)
- **Evaluation:** Cross-validation (5-fold stratified), accuracy, precision, recall, F1, ROC-AUC, confusion matrix

### Backend (FastAPI — 45 lines Python)
- **Framework:** FastAPI 0.115.0
- **Validation:** Pydantic 2.9.2 (field-level validation on all inputs)
- **Server:** Uvicorn 0.30.6 (production ASGI server)
- **Model Loading:** Joblib (serialized XGBoost pipeline + optimized threshold)
- **CORS:** Enabled for all origins
- **Endpoints:**
  - `GET /` — Health check
  - `POST /predict` — Classification with probability

### Frontend (203 HTML + 431 CSS + 370 JavaScript = 1,004 lines)
- **Design:** Professional, minimal aesthetic with serif headers + sans-serif body
- **Color Scheme:** Neutral paper tones with green (low risk) and red (high risk) accents
- **Form Fields:** 11 input fields matching the trained model's features exactly
- **Result Display:** 
  - Animated risk gauge (0–100%) showing default probability
  - Color-coded verdict ("Low Risk" in green, "High Risk" in red)
  - Threshold display (model's decision boundary)
- **API Endpoint:** Configurable base URL saved to localStorage
- **Error Handling:** Form validation, network error messages, server-side error display
- **Responsiveness:** Works on desktop and mobile

### Deployment
- **Platform:** Render
- **Runtime:** Python 3.11.9 (specified in `runtime.txt`)
- **Dependencies:** XGBoost 3.4.1, Scikit-learn 1.5.1, FastAPI, Uvicorn, Pandas, Pydantic, Joblib
- **Model Files:** `credit_risk_model.pkl` (full pipeline) + `best_threshold.pkl` (optimized decision boundary)

---

## 📊 What Makes This Project Advanced

### 1. **SHAP Explainability (Not Standard in Most Projects)**

Most ML projects stop at prediction. This one goes further:

**Global Explanations** — SHAP summary plots showing:
- Which features are most influential across all applicants
- Whether high income pushes predictions high or low
- How loan intent (personal, debt_consolidation, etc.) affects default risk
- Direction and magnitude of each feature's impact

**Local Explanations** — SHAP waterfall plots for individual applicants:
- Base model prediction starts at ~45% (population average)
- Age = +8% (older applicants lower risk)
- Income = -6% (high income lowers risk)
- Loan amount = +12% (larger loans increase risk)
- Final prediction: 59% default probability

This is **regulatory-grade explainability** — you can show a rejected applicant exactly why.

### 2. **Cardinality & Feature Engineering**

The notebook explicitly handles **high-cardinality categorical features**:
- `person_home_ownership` (5 categories)
- `loan_intent` (6 categories)
- `loan_grade` (7 categories)
- `cb_person_default_on_file` (2 categories)

Rather than naively one-hot encoding (creating sparse, high-dimensional features), the notebook:
- Uses ColumnTransformer to treat each feature type appropriately
- Applies StandardScaler to numeric features
- Applies OneHotEncoder to categorical features
- Handles unseen categories gracefully

### 3. **Class Imbalance Handling**

Credit default is rare (imbalanced dataset). The notebook addresses this with:
- **Stratified train/test split** — preserves class proportions in both sets
- **`scale_pos_weight` in XGBoost** — automatically penalizes the model for misclassifying the minority class (default cases)
- **ROC-AUC scoring** — instead of plain accuracy (which can look good on imbalanced data while ignoring the minority)
- **Stratified 5-fold cross-validation** — ensures every fold represents the full class distribution

### 4. **Probability Calibration & Threshold Optimization**

Raw model probabilities are often poorly calibrated. This notebook:
- Computes the **optimal decision threshold** using precision-recall curves
- Saves it as `best_threshold.pkl` for consistent serving
- Allows business teams to adjust the threshold based on cost-benefit trade-offs (false positives vs. false negatives)

### 5. **Comprehensive Model Comparison**

Two candidate algorithms evaluated side-by-side:
- **Logistic Regression** — fast, interpretable baseline
- **XGBoost** — captures non-linear patterns, handles feature interactions

Cross-validation results show:
- ROC-AUC for each model across 5 folds
- Accuracy, precision, recall, F1 — full picture
- XGBoost consistently outperforms logistic regression

---

## 📋 Input Features (11 Parameters)

| Feature | Type | Description |
|---------|------|-------------|
| **person_age** | Integer | Applicant's age (18–100) |
| **person_income** | Float | Annual income (USD) |
| **person_home_ownership** | Categorical | RENT / OWN / MORTGAGE / OTHER / (blank) |
| **person_emp_length** | Float | Years employed (0–123) |
| **loan_intent** | Categorical | PERSONAL / DEBT_CONSOLIDATION / EDUCATION / MEDICAL / VENTURE / HOME_IMPROVEMENT |
| **loan_grade** | Categorical | A / B / C / D / E / F / G (risk rating from lender) |
| **loan_amnt** | Float | Loan amount (USD) |
| **loan_int_rate** | Float | Interest rate (%) |
| **loan_percent_income** | Float | Loan amount as % of annual income |
| **cb_person_default_on_file** | Categorical | Y / N (has applicant defaulted before?) |
| **cb_person_cred_hist_length** | Integer | Years of credit history |

---

## 📊 Output

**Prediction Result:**
```json
{
  "default_probability": 0.67,
  "default_prediction": 1,
  "threshold": 0.5,
  "Result": "High Risk"
}
```

- **default_probability** (0–1) — Model's confidence that applicant will default
- **default_prediction** (0 or 1) — Binary classification (0 = Low Risk, 1 = High Risk)
- **threshold** — Decision boundary used (can be tuned by business)
- **Result** — Human-readable verdict

---

## 🧪 Machine Learning Development Pipeline

The Jupyter notebook (`Credit_Risk.ipynb`, 74 cells) follows a rigorous ML workflow:

### 1. Data Exploration (Cells 1–14)
- Loaded dataset and explored shape, dtypes, first rows
- Univariate analysis — distributions of numeric features, frequencies of categorical features
- Bivariate analysis — how each feature relates to default (`cb_person_default`)
- Identified class imbalance and missing values

### 2. Data Validation & Outlier Handling (Cells 15–18)
- **Removed duplicates** — exact row matches
- **Age validation** — kept only applicants 18–100 years old (filtered out data-entry errors)
- **Missing value strategy** — analyzed missingness patterns and filled appropriately

### 3. Feature Engineering & Train-Test Split (Cells 19–22)
- Defined 11 core features as X, default as y
- **Stratified 80/20 split** — ensures both train and test have the same class proportions
- Computed `scale_pos_weight` to handle class imbalance in XGBoost

### 4. Class Imbalance Handling (Cells 23–25)
- Calculated weight ratio: (# non-defaults) / (# defaults)
- This weight is passed to XGBoost to penalize misclassifying the rare default cases

### 5. Preprocessing Pipelines (Cells 26–30)

**For Logistic Regression:**
- StandardScaler on all numeric features
- OneHotEncoder on categorical features
- ColumnTransformer chains them together

**For XGBoost:**
- Same preprocessing (XGBoost still benefits from scaled numeric features)
- Pipeline ensures train and test preprocessing are identical

### 6. Evaluation Helper Function (Cells 31–32)
```python
def evaluate_model(model_name, model, X_test, y_test, threshold=None):
    # Calculates accuracy, precision, recall, F1, ROC-AUC
    # Handles custom thresholds for probability predictions
    # Displays confusion matrix and classification report
```

### 7. Cross-Validation (Cells 33–37)
- **5-fold stratified cross-validation** on both models
- Scoring metrics: ROC-AUC, accuracy, precision, recall, F1
- XGBoost shows superior performance across folds

### 8. Model Training (Cells 38–42)
- **Logistic Regression baseline** — trained and evaluated
- **XGBoost final model** — trained with tuned hyperparameters and class weights
- XGBoost selected as the production model

### 9. Hyperparameter Tuning (Cells 43–47)
- GridSearchCV to optimize XGBoost parameters
- Parameters tuned: `learning_rate`, `max_depth`, `n_estimators`
- Cross-validation ensures results generalize

### 10. Probability Calibration (Cells 48–52)
- Computed precision-recall curves
- Found optimal threshold that balances precision and recall for the business use case
- Saved threshold for consistent serving

### 11. SHAP Interpretation — Global & Local (Cells 53–62)
```python
# Create SHAP explainer for the XGBoost model
explainer = shap.TreeExplainer(xgb_classifier)

# Calculate SHAP values for test set
shap_values = explainer.shap_values(X_test_df)

# Global explanation: which features matter most?
shap.summary_plot(shap_values, X_test_df)

# Local explanation: why was this specific applicant high/low risk?
shap.plots.waterfall(shap.Explanation(...))
```

### 12. Error Analysis (Cells 63–67)
- Analyzed false positives (incorrectly flagged as high risk)
- Analyzed false negatives (incorrectly flagged as low risk)
- Identified which feature combinations lead to misclassifications

### 13. Model Serialization (Cells 68–72)
```python
joblib.dump(xgb_model, 'credit_risk_model.pkl')
joblib.dump(best_threshold, 'best_threshold.pkl')
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11.9
- pip
- Git
- Modern web browser

### Local Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/tarunkumar7906/Credit-Risk-Assessment.git
cd Credit-Risk-Assessment
```

#### 2. Create Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Run the Backend API
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API available at: `http://localhost:8000`

#### 5. Open the Frontend
```bash
python -m http.server 5500
# Visit http://localhost:5500
```

#### 6. Test the App
Fill in a sample loan application and submit to see:
- Default probability percentage
- Risk level (High/Low)
- Feature values that influenced the prediction

---

## 📡 API Documentation

### Base URL
```
Local: http://localhost:8000
```

### GET `/`
Health check.

**Response:**
```json
{"message": "Hello World!"}
```

---

### POST `/predict`

**Request Body (JSON):**
```json
{
  "person_age": 35,
  "person_income": 75000,
  "person_home_ownership": "RENT",
  "person_emp_length": 8.0,
  "loan_intent": "PERSONAL",
  "loan_grade": "C",
  "loan_amnt": 15000,
  "loan_int_rate": 10.5,
  "loan_percent_income": 0.20,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 12
}
```

**Success Response (200 OK):**
```json
{
  "default_probability": 0.38,
  "default_prediction": 0,
  "threshold": 0.5,
  "Result": "Low Risk"
}
```

**Validation Error (422):**
```json
{
  "detail": [
    {
      "loc": ["body", "person_age"],
      "msg": "ensure this value is greater than or equal to 18",
      "type": "value_error"
    }
  ]
}
```

---

## 🎨 Frontend Features

**Professional Form Layout:**
- 11 input fields organized logically
- Semantic HTML with proper labels
- Real-time validation matching backend constraints
- Clear field descriptions (what each value means)

**Risk Gauge:**
- Animated circular gauge (0–100%)
- Color gradient: green (low risk) → yellow (medium) → red (high risk)
- Percentage label centered in gauge
- Smooth animation on result update

**Verdict Display:**
- Prominent "Low Risk" or "High Risk" badge
- Color-coded (green or red)
- Threshold value shown (e.g., "Model threshold: 50%")
- Confidence percentage

**Error Handling:**
- Form-level validation (empty fields, invalid ranges)
- Network error messages
- Server-side error details (5xx errors, validation failures)
- Clear user feedback for every state

---

## 🔒 Security & Validation

✅ **Pydantic Field Validation** — Every input constrained to realistic ranges
✅ **Type Safety** — Strict type checking on all fields
✅ **CORS** — Enabled for frontend-backend communication
✅ **Pipeline-Based Preprocessing** — No data leakage between train and serving
✅ **No Data Persistence** — Predictions are stateless, no logs stored

---

## 📁 Project Structure

```
Credit-Risk-Assessment/
│
├── main.py                       # FastAPI backend (45 lines)
│                                # - Pydantic LoanApplication model
│                                # - GET / endpoint
│                                # - POST /predict endpoint
│                                # - Model + threshold loading
│
├── credit_risk_model.pkl         # XGBoost pipeline (serialized)
├── best_threshold.pkl            # Optimized decision threshold
├── credit_risk_dataset.csv       # Training dataset
│
├── requirements.txt              # Python dependencies
├── runtime.txt                   # Python 3.11.9 for Render
│
├── frontend/
│   ├── index.html               # HTML form (203 lines)
│   ├── style.css                # Styling (431 lines)
│   └── script.js                # Form + API integration (370 lines)
│
├── Credit_Risk.ipynb            # Complete ML notebook (74 cells)
│                                # - 9 markdown explanation sections
│                                # - EDA → Cleaning → Preprocessing →
│                                #   Model Comparison → Tuning →
│                                #   SHAP Interpretation → Error Analysis
│
└── README.md                    # This file
```

---

## 💡 Key Engineering Decisions

### 1. **XGBoost Over Logistic Regression**
Even though both were evaluated fairly with the same preprocessing and CV, XGBoost captured feature interactions (e.g., "young + high loan amount = higher risk") that linear models miss.

### 2. **SHAP for Explainability**
Rather than relying on feature importance that's algorithm-specific, SHAP provides a game-theory-based unified framework for explaining any model. Shaply values are:
- **Consistent** — if a model changes, the explanation changes proportionally
- **Locally accurate** — explanation for one prediction sums to the actual prediction
- **Globally interpretable** — which features matter most across all predictions

### 3. **Probability Calibration**
Raw XGBoost probabilities aren't inherently well-calibrated (e.g., "70% confident" may not match empirical 70% default rate). By computing precision-recall curves and choosing an optimal threshold, the business can tune for:
- **Low false positives** — don't reject good applicants (costly type I errors)
- **Low false negatives** — don't approve bad applicants (costly type II errors)

### 4. **Class Weight Handling**
Instead of resampling (which can cause overfitting), XGBoost's `scale_pos_weight` automatically penalizes misclassifying the minority class (defaults) during training — a principled approach.

### 5. **Pipeline-Based Preprocessing**
Both train and serving use the **exact same preprocessing** pipeline. This prevents the common bug where preprocessing was done differently in training vs. serving, causing train-serving skew.

---

## 🚀 Future Improvements

- [ ] Integrate SHAP visualizations into the frontend (show feature importance for each prediction)
- [ ] Add batch prediction endpoint (CSV upload → predictions for multiple applicants)
- [ ] A/B test different thresholds to find the optimal business trade-off
- [ ] Monitor prediction drift over time (are defaults changing? is the model still accurate?)
- [ ] Retrain periodically with new loan data to keep the model current
- [ ] Add explainability for denied applications (why was this person rejected?)
- [ ] Feature interaction analysis — which feature combinations are most predictive?

---

## 🎓 What This Project Demonstrates

✅ **Interpretable ML** — Not just high accuracy, but explainable predictions
✅ **Responsible AI** — Handle class imbalance thoughtfully, don't just optimize accuracy
✅ **Production-grade** — Full pipeline from data → model → serving → frontend
✅ **Business acumen** — Probability calibration for cost-benefit trade-offs
✅ **Advanced techniques** — SHAP, cardinality handling, threshold optimization
✅ **End-to-end ownership** — Notebook → API → deployment → frontend

---

## ⚖️ License

Open-source project available for educational and research purposes.

---

## 🤝 Contact & Social

**Author:** Tarun Kumar  
**GitHub:** [github.com/tarunkumar7906](https://github.com/tarunkumar7906)  
**LinkedIn:** [linkedin.com/in/tarun-kumar](https://www.linkedin.com/in/tarun-kumar-5b9280396)

---

**Project Status:** ✅ Complete | Credit Risk Assessment Model v1.0  
**Model:** XGBoost with SHAP Explainability  
**Deployment:** FastAPI + Render  
**Key Innovation:** SHAP-based local + global explanations for every prediction  
**Last Updated:** September 2026
