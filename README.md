# 🎣 AI-Powered Phishing Detector

![Build Status](https://img.shields.io/github/actions/workflow/status/credkellar-boop/Ai-Phishing-Detector/ci.yml?branch=main)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)

An AI-driven web application that detects phishing attempts in emails and text messages using Natural Language Processing (NLP) and Machine Learning.

## Features

* **Machine Learning:** Uses a Random Forest Classifier trained on text data. *(Note: Code utilizes an ensemble voting classifier incorporating Random Forest, Gradient Boosting, and SVC).*
* **NLP:** Utilizes TF-IDF vectorization to extract contextual features from text.
* **Web Interface:** Clean, interactive UI built with Streamlit.

## Project Structure

```text
Ai-Phishing-Detector/
├── .github/
│   └── workflows/
│       └── ci.yml             # Continuous Integration pipeline configuration
├── model/                     # Directory generated after running train_model.py
│   ├── phishing_model.pkl     # Serialized machine learning ensemble model
│   └── vectorizer.pkl         # Serialized TF-IDF vectorizer
├── .gitignore                 # Specifies intentionally untracked files to ignore
├── README.md                  # Project documentation
├── app.py                     # Main Streamlit web application script
├── requirements.txt           # List of project dependencies
└── train_model.py             # Script to fetch data, train, and save the models
