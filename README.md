# ❤️ Heart Disease Prediction Model

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Model-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

---
<img width="1920" height="1110" alt="image" src="https://github.com/user-attachments/assets/b80b41bb-46e7-4fd5-937e-22afbb87e5ee" />


## 📌 Overview

The **Heart Disease Prediction Model** is a completed Machine Learning project designed to predict the likelihood of heart disease using clinical and patient health data.

The project combines **data preprocessing, exploratory analysis, model training, performance comparison, and web-based prediction** into an end-to-end Machine Learning application.

> ⚠️ **Medical Disclaimer:** This project is intended for educational and research purposes only. It is not a substitute for professional medical diagnosis, clinical judgment, or treatment.

---

## 🎯 Project Goal

The primary goal of this project is to build a Machine Learning system capable of analyzing patient health indicators and generating a prediction related to heart disease.

The project demonstrates how Machine Learning can be applied to healthcare datasets to support **data-driven analysis and early risk assessment**.

---

## 🩺 Prediction Inputs

The web application collects the following patient information:

| Feature                     | Description                                    |
| --------------------------- | ---------------------------------------------- |
| **Age**                     | Patient's age in years                         |
| **Sex**                     | Biological sex of the patient                  |
| **Chest Pain Type**         | Type of chest pain experienced                 |
| **Resting Blood Pressure**  | Blood pressure measured while resting          |
| **Cholesterol**             | Total blood cholesterol level                  |
| **Fasting Blood Sugar**     | Whether fasting blood sugar is above 120 mg/dL |
| **Resting ECG**             | Result of the resting electrocardiogram        |
| **Max Heart Rate**          | Maximum heart rate achieved during exercise    |
| **Exercise-Induced Angina** | Whether exercise produces chest pain           |
| **Oldpeak (ST Depression)** | ST-segment depression induced by exercise      |
| **ST Slope**                | Slope of the peak exercise ST segment          |

### Input Categories

The application provides categorical selections for:

**Sex**

* Male
* Female

**Chest Pain Type**

* Typical Angina
* Atypical Angina
* Non-Anginal Pain
* Asymptomatic (No Pain)

**Fasting Blood Sugar**

* Yes (> 120 mg/dL)
* No (≤ 120 mg/dL)

**Resting ECG**

* Normal
* ST-T Wave Abnormality
* Left Ventricular Hypertrophy

**Exercise-Induced Angina**

* Yes
* No

**ST Slope**

* Upsloping (Normal)
* Flat
* Downsloping

---

## 🚀 Features

* 🧹 **Data preprocessing and cleaning**
* 📊 **Exploratory data analysis and visualization**
* 🤖 **Multiple Machine Learning models**
* ⚖️ **Model performance comparison**
* 📈 **Evaluation using classification metrics**
* 💾 **Model serialization using Pickle / Joblib**
* 🖥️ **Interactive web-based prediction interface**
* ❤️ **Heart disease prediction from user-provided clinical data**

---

## 🧠 Machine Learning Models

Several classification algorithms were implemented and compared:

### Logistic Regression

A linear classification algorithm used as a baseline model for predicting the probability of heart disease.

### Decision Tree

A tree-based model that makes predictions through a sequence of feature-based decisions.

### Random Forest ⭐

An ensemble learning method that combines multiple decision trees to improve prediction performance and robustness.

**Random Forest is used as the primary prediction model in the completed application.**

### Gradient Boosting

An ensemble technique that builds models sequentially, with each new model attempting to improve the errors of previous models.

### Support Vector Machine (SVM)

A classification algorithm that identifies a decision boundary capable of separating different classes within the dataset.

---

## 🔬 Machine Learning Workflow

The project follows an end-to-end Machine Learning workflow:

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Data Preprocessing
     ↓
Exploratory Data Analysis
     ↓
Feature Selection / Preparation
     ↓
Train-Test Split
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Model Comparison
     ↓
Primary Model Selection
     ↓
Model Serialization
     ↓
Web Application
     ↓
Heart Disease Prediction
```

---

## 📊 Model Evaluation

The implemented models were evaluated and compared using classification performance metrics.

Evaluation focuses on measuring how effectively each model distinguishes between patients with and without the target condition.

Common evaluation metrics used in the project include:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**

The models were compared to identify the most suitable model for deployment in the application.

---

## 🖥️ Web Application

The trained Machine Learning model is integrated into an interactive web interface.

Users can enter patient information through the provided form, including:

* Age
* Sex
* Chest Pain Type
* Resting Blood Pressure
* Cholesterol
* Fasting Blood Sugar
* Resting ECG
* Max Heart Rate
* Exercise-Induced Angina
* Oldpeak
* ST Slope

The application processes the submitted information and passes the resulting feature vector to the trained Machine Learning model to generate a prediction.

![Heart Disease Prediction Interface](https://github.com/user-attachments/assets/8f965ab5-f45f-47a0-9819-f9e3a5cd5715)

---

## 🛠️ Technology Stack

### Programming Language

* Python 3.10+

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn

### Data Visualization

* Matplotlib
* Seaborn

### Model Persistence

* Pickle
* Joblib

### Development

* Visual Studio Code
* Git
* GitHub

---

## 📌 Prerequisites

Before running the project, make sure the following are installed:

* Python 3.10 or later
* Git
* Visual Studio Code *(recommended)*

You should also have:

* A GitHub account
* Access to the project repository

---

## 📥 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/YoussefTaha39/Heart-Disease-Prediction-Model.git
cd Heart-Disease-Prediction-Model
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🌿 Git Workflow

The project uses the `dev` branch for development.

### Switch to the development branch

```bash
git checkout dev
```

### Pull the latest changes

```bash
git pull origin dev
```

### After making changes

```bash
git add .
git commit -m "Your message here"
git push origin dev
```

Use clear and descriptive commit messages to make project history easier to understand.

---

## 📂 Project Structure

A typical project structure can be organized as follows:

```text
Heart-Disease-Prediction-Model/
│
├── data/
│   └── dataset.csv
│
├── models/
│   └── trained_model.pkl
│
├── static/
│   └── ...
│
├── templates/
│   └── ...
│
├── app.py
├── requirements.txt
├── README.md
└── ...
```

The exact structure may vary depending on the final implementation.

---

## ✅ Project Status

**Status: Completed ✅**

The Machine Learning pipeline, model comparison, trained prediction model, serialization process, and interactive web interface have been implemented as part of the completed project.

---

## ❤️ Project Vision

The project demonstrates the potential of Machine Learning in healthcare-related applications by transforming clinical data into actionable predictive information.

The broader vision is to explore how **AI-powered healthcare systems** can support faster, more accessible, and data-driven analysis while keeping professional medical expertise at the center of real-world decision-making.

---
