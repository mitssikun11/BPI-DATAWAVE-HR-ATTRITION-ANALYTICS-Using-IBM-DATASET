# IBM HR Analytics - BPI DATA Wave Competition

## 🎯 Project Overview

This project is specifically designed for the **BPI DATA Wave** competition using the **IBM HR Analytics Employee Attrition** dataset. The goal is to build advanced machine learning models to predict employee attrition, helping organizations identify and retain at-risk employees.

## 📊 Dataset Information

**Dataset**: `WA_Fn-UseC_-HR-Employee-Attrition.csv`
- **Size**: 1,470 employees
- **Features**: 35 variables including demographic, job-related, and satisfaction metrics
- **Target**: Employee Attrition (Yes/No)
- **Attrition Rate**: ~16% (imbalanced dataset)

### Key Features:
- **Demographic**: Age, Gender, Education, Marital Status
- **Job-related**: Department, Job Role, Job Level, Years at Company
- **Compensation**: Monthly Income, Salary Hike, Stock Options
- **Satisfaction**: Job Satisfaction, Environment Satisfaction, Work-Life Balance
- **Behavioral**: Overtime, Business Travel, Distance from Home

## 📁 Project Structure

```
ibm_hr_analytics_project/
├── data/                           # Dataset files
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv  # Main dataset
├── models/                         # Trained models and artifacts
├── notebooks/                      # Analysis notebooks
├── src/                           # Source code modules
│   ├── competition_data_loader.py  # Generic data loader
│   └── data_loader.py             # Original data loader
├── utils/                         # Utility functions
│   └── visualization.py           # Visualization tools
├── submissions/                   # Competition outputs
│   ├── hr_best_model.pkl         # Best HR model
│   ├── hr_predictions.csv        # HR predictions
│   ├── hr_model_performance.csv  # HR model comparison
│   └── hr_feature_importance.csv # HR feature analysis
├── docs/                         # Documentation
│   └── COMPETITION_GUIDE.md     # Competition guide
├── requirements.txt              # Python dependencies
├── hr_competition_trainer.py    # HR-specific trainer
├── competition_trainer.py        # Generic trainer
├── train_model.py               # Original training script
├── test_hr_pipeline.py          # HR pipeline test
└── README.md                    # Project overview
```

## 🚀 Quick Start

### 1. Installation
```bash
# Install dependencies
python install_dependencies.py

# Or manually
pip install -r requirements.txt
```

### 2. Test HR Pipeline
```bash
# Test the HR pipeline with actual dataset
python test_hr_pipeline.py
```

### 3. Run Full HR Pipeline
```bash
# Run the complete HR competition pipeline
python hr_competition_trainer.py
```

### 4. Check Results
After running the pipeline, check the `submissions/` directory for:
- `hr_best_model.pkl` - Best performing HR model
- `hr_predictions.csv` - HR attrition predictions
- `hr_model_performance.csv` - HR model comparison
- `hr_feature_importance.csv` - HR feature analysis
- Various HR-specific visualization plots

## 🔧 Key Components

### HR Competition Trainer (`hr_competition_trainer.py`)

**Specialized Features:**
- **HR Dataset Loading**: Direct integration with IBM HR dataset
- **Attrition Prediction**: Optimized for employee turnover prediction
- **Class Imbalance Handling**: Special techniques for imbalanced attrition data
- **HR-Specific Metrics**: Focus on recall for identifying at-risk employees
- **Feature Engineering**: HR domain-specific preprocessing

**Algorithms Optimized for HR:**
- **Random Forest**: With class balancing for imbalanced data
- **XGBoost**: With scale_pos_weight for attrition prediction
- **LightGBM**: With balanced class weights
- **CatBoost**: With class weights for minority class
- **Logistic Regression**: With balanced class weights
- **HR Ensemble**: Voting classifier combining top models

## 📊 HR-Specific Metrics

The HR competition uses a specialized scoring system:

```
HR Competition Score = (ROC_AUC × 0.3) + (F1_Score × 0.3) + (Recall × 0.4)
```

**Why focus on Recall?**
- **Business Impact**: Missing an employee who will leave (false negative) is more costly than incorrectly flagging someone (false positive)
- **Retention Strategy**: Better to identify more potential leavers and implement retention strategies
- **Cost-Benefit**: False positives are cheaper than false negatives in HR context

## 🎯 HR Competition Strategy

### 1. Data Preprocessing
- **Target Encoding**: Convert Yes/No to 1/0 for attrition
- **Missing Values**: Median imputation for numerical, mode for categorical
- **Categorical Encoding**: Label encoding for all categorical variables
- **Feature Scaling**: StandardScaler for numerical features
- **Class Balancing**: Special handling for imbalanced attrition data

### 2. Model Selection
- **Baseline**: Logistic Regression with balanced weights
- **Tree-based**: Random Forest, XGBoost, LightGBM, CatBoost with class balancing
- **Ensemble**: Voting classifier with top 3 models
- **Optimization**: Focus on recall and ROC AUC

### 3. Feature Importance
- **HR Domain Knowledge**: Understanding which factors drive attrition
- **Business Insights**: Identifying actionable retention strategies
- **Model Interpretability**: Clear feature importance for HR decisions

### 4. Evaluation Strategy
- **Cross-validation**: 5-fold stratified CV
- **Holdout set**: 20% for final evaluation
- **Multiple metrics**: ROC AUC, F1, Precision, Recall
- **Business Focus**: Emphasis on recall for retention strategies

## 📈 HR-Specific Features

### Class Imbalance Handling
The HR dataset has ~16% attrition rate, requiring special handling:

```python
# XGBoost with class balancing
xgb_model = XGBClassifier(
    scale_pos_weight=3,  # Handle class imbalance
    random_state=42
)

# Random Forest with balanced weights
rf_model = RandomForestClassifier(
    class_weight='balanced',
    random_state=42
)
```

### HR-Specific Scoring
```python
# HR scoring emphasizes recall
hr_score = (roc_auc * 0.3) + (f1_score * 0.3) + (recall * 0.4)
```

### Feature Importance Analysis
- Automatic identification of key attrition factors
- Business insights for retention strategies
- Actionable recommendations for HR teams

## 📋 HR Submission Requirements

### Required Files
1. **`hr_predictions.csv`**: HR attrition predictions
   - Format: `id, attrition_probability`
   - Values: Probability scores (0-1)

2. **`hr_best_model.pkl`**: Trained HR model
   - Serialized model for deployment
   - Can be loaded with `joblib.load()`

3. **`hr_model_performance.csv`**: HR model comparison
   - All models with their metrics
   - For analysis and documentation

### Optional Files
- `hr_feature_importance.csv`: HR feature analysis
- HR-specific visualization plots (PNG format)
- Detailed HR analysis reports

## 🔍 HR Data Analysis Workflow

### 1. Data Exploration
```python
# Load and analyze HR data
trainer = HRCompetitionTrainer()
data = trainer.load_hr_data()
trainer.analyze_hr_data()
```

### 2. Model Training
```python
# Train HR models
trainer.preprocess_hr_data()
trainer.train_hr_models()
```

### 3. Ensemble Creation
```python
# Create HR ensemble model
trainer.create_hr_ensemble()
```

### 4. Report Generation
```python
# Generate comprehensive HR report
trainer.generate_hr_report()
```

## 🛠️ HR Customization

### Adding HR-Specific Features
```python
# In hr_competition_trainer.py
def create_hr_features(self, data):
    # Add HR-specific feature engineering
    data['tenure_ratio'] = data['YearsAtCompany'] / data['Age']
    data['promotion_rate'] = data['YearsSinceLastPromotion'] / data['YearsAtCompany']
    return data
```

### Custom HR Metrics
```python
def hr_business_metric(self, y_true, y_pred, y_pred_proba):
    # Implement HR-specific business metrics
    # e.g., cost of false negatives vs false positives
    return business_score
```

## 📊 HR Performance Monitoring

### Model Comparison
The system automatically compares all models and selects the best one based on:
- ROC AUC score
- F1 score
- Recall score (emphasized for HR)
- Custom HR competition scoring

### HR Visualization
- HR model performance comparison plots
- ROC curves for all HR models
- HR feature importance charts
- Attrition prediction analysis

## 🚨 HR Troubleshooting

### Common HR Issues

1. **Class Imbalance**
   - Use class balancing techniques
   - Focus on recall metrics
   - Consider data augmentation

2. **Feature Engineering**
   - Create HR-specific features
   - Consider domain knowledge
   - Handle categorical variables properly

3. **Model Interpretability**
   - Use tree-based models for feature importance
   - Provide business insights
   - Focus on actionable recommendations

### HR Debug Mode
```python
# Enable verbose output for HR pipeline
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📞 HR Support

For HR-specific questions:
- Check the `README.md` for general information
- Review `PROJECT_IDEAS.md` for advanced features
- Examine the HR trainer code for implementation details

## 🎯 HR Competition Tips

1. **Focus on Recall**: Missing attrition cases is more costly
2. **Handle Imbalance**: Use class balancing techniques
3. **Feature Engineering**: Create HR-specific features
4. **Business Insights**: Provide actionable recommendations
5. **Model Interpretability**: Explain predictions to HR teams
6. **Cost-Benefit**: Consider false positive vs false negative costs
7. **Retention Strategy**: Focus on factors that can be influenced
8. **Validation**: Use cross-validation consistently

## 📈 HR Next Steps

1. **Run HR Pipeline**: Execute the complete HR analysis
2. **Review Results**: Analyze model performance and feature importance
3. **Business Insights**: Identify key attrition factors
4. **Retention Strategy**: Develop actionable recommendations
5. **Model Deployment**: Prepare for production use
6. **Monitoring**: Set up ongoing model monitoring

## 🏆 HR Business Impact

This HR analytics solution provides:

- **Predictive Insights**: Identify employees at risk of leaving
- **Retention Strategies**: Focus on modifiable factors
- **Cost Savings**: Reduce recruitment and training costs
- **Employee Satisfaction**: Proactive retention improves morale
- **Data-Driven Decisions**: Evidence-based HR strategies

---

**Ready to predict employee attrition and improve retention strategies! 🚀** 