# Fake News Detector

A machine learning-based web application that detects whether a news article is **Real** or **Fake** using Natural Language Processing (NLP).

## Project Description

The Fake News Detector uses a trained ML model to classify news text as either **fake** or **real**. It employs **TF-IDF vectorization** and **Logistic Regression** to analyze and predict the credibility of the input content.
This tool is built to help users identify misinformation and promote awareness in a digital age where fake news spreads rapidly.

## Features
-  Real-time detection of fake news
-  Trained using a labeled dataset of real and fake news articles
-  Achieved high accuracy (~98%)
-  Web interface using Streamlit
-  Supports manual input of any news text
-  Easily extendable for future improvements

## Tech Stack
- **Python**
- **Pandas, NumPy**
- **Scikit-learn**
- **NLTK**
- **Streamlit**
- **Pickle**

## Model Training
1. The model is trained using a merged dataset of real and fake news.
2. Text is preprocessed and vectorized using **TF-IDF**.
3. A **Logistic Regression** classifier is trained and saved using Pickle.

## Running the App
Install dependencies:
pip install -r requirements.txt

## Run the Streamlit app:
cd App 
- python -m streamlit run streamlit_app.py

## Model Performance
Accuracy: 98.5%
