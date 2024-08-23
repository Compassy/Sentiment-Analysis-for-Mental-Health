
# Sentiment Analysis for Mental Health

## Overview

The project uses a dataset of mental health statements tagged with various mental health statuses on [kaggle](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health). It includes data preprocessing, exploratory data analysis, model training, and evaluation.

## Steps

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Text preprocessing using spaCy
- TF-IDF vectorization
- Oversampling using SMOTE
- Multiple classification models comparison
- Model evaluation and fine-tuning
- Model Saving

## Requirements

- Python 3.11.x
- Libraries: numpy, pandas, scikit-learn, spacy, imbalanced-learn, matplotlib, seaborn, tqdm, pickle, scipy

## Installation

1. Clone this repository
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```
3. Download the spaCy English model:
   ```
   python -m spacy download en_core_web_sm
   ```

## Usage

1. Run the Jupyter notebook `Sentiment_Analysis.ipynb`
2. The notebook will guide you through the entire process from data loading to model evaluation

## Models Evaluated

- Logistic Regression
- Decision Tree Classifier
- Extra Tree Classifier
- AdaBoost Classifier
- Random Forest Classifier
- Extra Trees Classifier
- Gradient Boosting Classifier
- Bagging Classifier
- SGD Classifier
- SVC
- MLP Classifier

## Fine-tuning

The project includes fine-tuning of the best-performing model using RandomizedSearchCV.

## Contributors
- Piyawat Nulek (piyawat.nulek@icloud.com)