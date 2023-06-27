# Bank Customer Churn Prediction

<img width="605" alt="Screenshot 2023-06-27 at 9 51 06 AM" src="https://github.com/SaloniJhalani/Bank-Customer-Churn-Prediction/assets/33859675/aebb2d8b-b78b-4bbe-8322-e005ed3132ec">

## Table of Contents
- [Project Overview](#project-overview)
- [Data Source](#data-source)
- [Website Link](#website-link)
- [Implementation Details](#implementation-details)
    - [Methods Used](#methods-used)
    - [Technologies](#technologies)
    - [Python Packages Used](#python-packages-used)
- [Steps Followed](#steps-followed)
- [Results and Evaluation Criterion](#results-and-evaluation-criterion)
- [Future Improvements](#future-improvements)

## Project Overview
This project aims to develop a machine learning model to predict bank customer churn. Customer churn, also known as customer attrition, refers to the phenomenon where customers stop doing business with a company or switch to a competitor. By identifying potential churners in advance, banks can take proactive measures to retain valuable customers and minimize revenue loss. This repository provides a comprehensive solution for predicting customer churn using various machine learning algorithms.

## Data Source
The dataset used for this project can be obtained from [here](https://www.kaggle.com/datasets/adammaus/predicting-churn-for-bank-customers).

It contains relevant customer information such as demographics, account activity and churn status. 

## Website Link

A web-based demonstration of this project can be accessed [here](https://bank-customer-churn-prediction.streamlit.app).

## Implementation Details

### Methods Used
* Machine Learning
* Data Visualization
* Exploratory Data Analysis
* Predictive Modeling

### Technologies
* Python
* Jupyter
* streamlit

### Python Packages Used
* Data Manipulation: Pandas, numpy
* Data Visualization: seaborn, matplotlib
* Data Preprocessing: imblearn
* Machine Learning: scikit, xgboost
  
## Steps Followed

1. **Data collection**: Obtained the customer churn dataset from Source Dataset Link.
2. **Exploratory Data Analysis (EDA)**: Performed the comprehensive exploratory analysis to gain insights into the dataset, identify patterns, correlations, and potential predictive features.
3. **Data Preprocessing**: Cleansed the data, handled categorical values, normalized the data, and addressed imbalanced data.
4. **Model Development**: Implemented and trained various machine learning algorithms such as logistic regression, decision trees, random forests, and XGB.
5. **Model Evaluation**: Assessed the performance of each model using metrics such as accuracy, precision, recall, F1-score. Selected the best-performing model for deploymet.
6. **Deployment**: Deployed the chosen churn prediction model as a standalone application to enable real-time predictions for new customer data.

## Results and Evaluation Criterion

Based on the evaluation results, the best-performing model was **XGBoost** which achieved an accuracy of 91% and F1-score of 91%. 

<img width="436" alt="Screenshot 2023-06-27 at 4 15 32 PM" src="https://github.com/SaloniJhalani/Bank-Customer-Churn-Prediction/assets/33859675/4d273530-aa25-443c-93d6-33d6d3904de4">

## Future Improvements

Here are some potential areas for future improvements in the project:

* Feature engineering: Explore additional features that may capture customer behavior more accurately.
* Model optimization: Fine-tune the model parameters to potentially improve performance.
* Real-time monitoring: Implement a system to monitor and retrain the model periodically to adapt to changing customer dynamics.
* User interface enhancement: Improve the user interface of the deployed application for better user experience.






