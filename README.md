# Startup Success Prediction 🚀

This repository contains the graduation project for the Cretiva NTI Track (Machine Learning for Data Science). The project aims to predict the success of startups based on historical funding, industry categories, and other features.

## Project Structure

```text
├── .gitignore                <- Files for git to ignore
├── README.md                 <- The top-level README for developers using this project.
├── requirements.txt          <- Dependencies to run the project
├── data/
│   ├── raw/                  <- Original immutable data
│   └── processed/            <- Cleaned datasets ready for modeling
├── notebooks/                <- Jupyter notebooks for EDA and ML
│   ├── 01_data_cleaning_and_feature_engineering.ipynb
│   └── 02_machine_learning_modeling.ipynb
├── models/                   <- Trained models, scalers, and config files
└── app/                      <- Streamlit application for deployment
    └── app.py
```

## Team Members & Tasks

### 1. Ziad (Data Analysis & Preprocessing)
- Exploratory Data Analysis (EDA)
- Data Cleaning & handling missing values
- Feature Engineering (e.g. `company_age`, `funding_duration`)
- Handling Outliers via Capping (Winsorization)
- Feature Encoding and Scaling
- **Deliverables**: Processed datasets in `data/processed/`, EDA notebook, and deployment artifacts in `models/`.

### 2. Mohamed (Machine Learning)
- Experimenting with 6-8 machine learning models (e.g. Random Forest, XGBoost, Logistic Regression)
- Hyperparameter Tuning & Cross Validation
- Model comparison and selection
- Model interpretability using SHAP / Feature Importance
- **Deliverables**: Trained `model.pkl` saved in `models/`, ML Notebook, and a comparison report.

### 3. Ahmed (Application & Presentation)
- Building a Streamlit web application (`app/app.py`)
- Allowing user input to predict a new startup's success
- Visualizing important features and insights
- **Deliverables**: Streamlit App, Live Demo, and Final Presentation.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run app/app.py
   ```
