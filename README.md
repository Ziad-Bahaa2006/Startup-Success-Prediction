<div align="center">

<img src="https://img.icons8.com/?size=256&id=46101&format=png" alt="Startup Success Predictor Logo" width="120">

# 🚀 Startup Success Predictor

**An End-to-End Machine Learning Solution for Predicting Startup Outcomes**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-009688?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)](#)

*Developed for the Cretiva NTI Track (Machine Learning for Data Science) Graduation Project.*

---
</div>

## 📖 Table of Contents
- [About the Project](#-about-the-project)
- [Key Features](#-key-features)
- [Architecture & Methodology](#-architecture--methodology)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [Usage & Web App](#-usage--web-app)
- [Team Contributions](#-team-contributions)
- [License](#-license)

---

## 💡 About the Project
Venture capital and startup investments involve high risk. The **Startup Success Predictor** is an intelligent web application designed to forecast whether a startup will eventually succeed (e.g., via acquisition or IPO) or fail. By analyzing historical funding data, industry categories, geographical locations, and operational timelines, our models provide actionable insights to potential investors and founders.

## ✨ Key Features
- **Comprehensive Data Pipeline**: Robust handling of missing values, outlier capping (Winsorization), and advanced feature engineering (e.g., `company_age`, `funding_duration`).
- **High-Performance Modeling**: Leverages state-of-the-art algorithms including Random Forest, XGBoost, and Logistic Regression with rigorous hyperparameter tuning.
- **Model Interpretability**: Integrates SHAP values to explain feature importance, bringing transparency to the AI's decision-making process.
- **Interactive Web Interface**: A sleek, user-friendly Streamlit dashboard allowing single-startup predictions and bulk CSV processing.

## 🏗 Architecture & Methodology
Our workflow strictly follows the standard Data Science Lifecycle:
1. **Data Acquisition & EDA**: Understanding the underlying distributions and relationships in the startup ecosystem.
2. **Preprocessing**: Feature encoding, scaling, and engineering derived metrics to maximize model performance.
3. **Model Selection**: Cross-validating multiple classifiers to prevent overfitting and select the most robust model.
4. **Deployment**: Packaging the finalized `.pkl` models and scalers into an interactive Streamlit application.

## 📂 Repository Structure

```text
├── data/
│   ├── raw/                  <- Original immutable datasets
│   └── processed/            <- Cleaned datasets ready for modeling
├── notebooks/                <- Jupyter notebooks for EDA and Model Training
│   ├── 01_data_cleaning_and_feature_engineering.ipynb
│   └── 02_machine_learning_modeling.ipynb
├── models/                   <- Pickled models, scalers, and config files
├── app/                      <- Streamlit application for deployment
│   └── app.py
├── AI Startup Success Predictor .pptx  <- Final Project Presentation
├── requirements.txt          <- Dependencies and libraries
├── .gitignore                <- Files ignored by Git
└── README.md                 <- Project documentation (You are here)
```

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
Make sure you have Python 3.8+ installed. It is highly recommended to use a virtual environment.

### Installation
1. **Clone the repository** (if applicable):
   ```bash
git clone https://github.com/Ziad-Bahaa2006/Startup-Success-Prediction.git
   cd Startup-Success-Prediction
   ```

2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage & Web App

To launch the interactive dashboard, simply run the Streamlit server from the root directory:

```bash
streamlit run app/app.py
```
> **Note**: The application will automatically open in your default web browser at `http://localhost:8501`. 

From the interface, you can:
- **Predict Single Startup**: Input parameters manually to see the predicted outcome.
- **Bulk Prediction**: Upload a CSV file (e.g., `test_bulk_startups.csv`) to process multiple startups simultaneously.

## 👥 Team Contributions

This project was a collaborative effort by our dedicated data science team. 

| Member | Focus Area | Key Deliverables |
| :--- | :--- | :--- |
| **Ziad Bahaa** | Data Analysis & Preprocessing | EDA, Outlier Handling (Winsorization), Feature Engineering (`company_age`), Scaling, Processed Datasets. |
| **Mohamed** | Machine Learning & Modeling | Algorithm Selection (RF, XGBoost), Hyperparameter Tuning, Cross-Validation, SHAP Interpretability, Final `model.pkl`. |
| **Ahmed** | Application & Deployment | Streamlit UI/UX Design, App Architecture (`app.py`), Integration of ML pipelines, Bulk Prediction features, Final Presentation. |

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.

---
<div align="center">
  <i>Built with ❤️ by the Cretiva NTI Data Science Team.</i>
</div>
