
# Smart Symptom Checker: AI-Powered Disease Prediction System"

![Facebook vs AdWords](https://static.scientificamerican.com/sciam/cache/file/01C9741F-6F6D-4882-8217D92370325AA7_source.jpg?crop=4%3A3%2Csmart&w=1200)

# Problem Statement
Many individuals experience symptoms but delay seeking medical attention due to lack of awareness or fear. Early detection of possible diseases based on symptoms can greatly improve treatment outcomes. However, identifying diseases accurately from multiple overlapping symptoms is complex and requires trained professionals or intelligent systems.



# 🌟 Project Objective

To build an AI-powered web application that:
- Accepts user-inputted symptoms from a list of 130+ options.
- Uses a trained machine learning model to predict the most likely disease.
- Educates users about their potential condition and encourages early consultation.

# 🏋️️Project Goals

- ✅ Build a disease classification model using ensemble and linear algorithms.
- ✅ Design a robust and clean UI for symptom input.
- ✅ Minimize false predictions by enforcing symptom thresholding.
- ✅ Allow the system to be deployed easily for public or institutional use.
- ✅ Handle imbalanced classes using ML techniques (e.g., One-vs-Rest, class weights).


# 🚀 Project Overview

This project is an AI-powered Disease Prediction System that uses machine learning to predict possible diseases based on user-reported symptoms. The application provides a visually appealing Flask-based frontend where users can select symptoms, and the backend intelligently processes these inputs to generate a prediction.

The core pipeline consists of the following components:

## 🔄 1. Data Ingestion
Loads and combines the provided Training.csv and Testing.csv files.
Cleans the column names (e.g., removes special characters, trims whitespace).
Saves preprocessed data into the Artifacts/ directory:
train.csv
test.csv
data.csv
## 🔧 2. Data Transformation (Preprocessing)
## Label Encoding:
- The target variable (prognosis) is encoded using LabelEncoder for multi-class classification.
- The trained encoder is saved as label_encoder.pkl.
## Feature Transformation:
- Identifies numeric and categorical columns (if any).
- SimpleImputer to fill missing values.
- StandardScaler for numeric features.
- OneHotEncoder for categorical features.
- A ColumnTransformer pipeline is saved as preprocessor.pkl.
- Combines transformed features with encoded labels into NumPy arrays (train_array and test_array).
## 🧠 3. Model Training & Evaluation
- Trains multiple classifiers using OneVsRestClassifier for multi-class safety:
- Logistic Regression (with class_weight='balanced')
- AdaBoost
- (Others like SVC, CatBoost, RandomForest optionally included)
- Uses GridSearchCV for hyperparameter tuning.
- Automatically selects and saves the best performing model as model.pkl based on test accuracy.

## 🎯 4. Prediction Pipeline
- User selects symptoms from the UI.
- A feature vector is generated (binary 0/1 for 132 symptoms).
- The model predicts the encoded disease label.
- The label is then decoded using the saved encoder (label_encoder.pkl) and returned as a human-readable disease name.
- Includes logic to avoid inaccurate predictions when too few symptoms are selected (e.g., < 3 symptoms).
## 🌐 5. Web Application
- Built using Flask + Bootstrap for responsive, user-friendly UI.
- Symptom selection form with sliders and categories result.


# 🧠 Key Features

- ✅ Symptom Selection: Over 130 symptoms available via toggle sliders
- ✅ Disease Prediction: Output based on ensemble model trained on medical records
- ✅ Thresholding: Requires at least 3 symptoms for valid prediction
- ✅ User Interface: Built with Bootstrap 5, fully responsive
- ✅ Model Training Pipeline: Easily retrainable with new data
- ✅ Logging and Debugging: Custom error handling and logging enabled
- ✅ One-vs-Rest Classification: Handles multiclass prediction safely
- ✅ Class Balancing: Uses class_weight='balanced' for better prediction fairness

# 🚀 Deliverables

- 🎯 AI-powered disease prediction web tool
- 📦 Trained ML model with 90%+ test accuracy
- 📁 Flask web interface with index, form, and result pages
- 📚 Modular ML pipeline scripts
- 📊 GridSearch-based model evaluator
- 📌 README and usage documentation




# 🧠 Tools, Libraries & Techniques Used

| Category         | Tool / Library                                                   |
| ---------------- | ---------------------------------------------------------------- |
| Frontend         | HTML, CSS, Bootstrap 5                                           |
| Backend          | Python, Flask                                                    |
| ML Algorithms    | LogisticRegression, AdaBoost, XGBoost, CatBoost, SVC             |
| Preprocessing    | scikit-learn (`LabelEncoder`, `OneHotEncoder`, `StandardScaler`) |
| Evaluation       | accuracy\_score, GridSearchCV                                    |
| Deployment Ready | Flask, Pickle                                                    |
| Logging          | Python `logging` module                                          |
| Environment      | Python 3.10+                                                     |



# 🔎 Key Insights & Best Results

| Model               | Train Accuracy | Test Accuracy |
| ------------------- | -------------- | ------------- |
| LogisticRegression  | 0.92         | 0.90      |
| AdaBoost            | 0.96          | 0.95        |




# 🚀 How to Deploy the Project Locally

This guide walks you through deploying the Multimodal Pneumonia Detection AI Tool (or any symptom-based AI health tool) from your GitHub repository.

## ✅ Prerequisites
Make sure the following are installed on your system:

- Python 3.8+
- Git
- pip (pip --version to check)
- A virtual environment tool like venv or conda
## 🧩 1. Clone the GitHub Repository
- git clone https://github.com/Prabh10p/Multimodal-Pneumonia-Detection-AI-Tool.git
- cd Multimodal-Pneumonia-Detection-AI-Tool
## 📦 2. Create and Activate Virtual Environment
- python -m venv venv


##  📜 3. Install Required Dependencies
- Install the necessary libraries using the requirements.txt:
- pip install -r requirements.txt


## 🖥️ 4. Run the Flask Web App
- Once the model is trained and artifacts are ready, run the Flask app= python app.py
## 🌐 5. Access the Web App
- After launching, visit this URL in your browser:

- http://127.0.0.1:5000/predict



# 📌 Future Enhancements

 - Show description & treatment of predicted disease
 - Add confidence score with model prediction
 - Allow PDF download of results
 - Add multilingual support (Punjabi, Hindi, Spanish)
 - Deploy on HuggingFace Spaces or Streamlit Cloud
 - Integrate user data storage for medical history

# 👨‍💻 Author

**Prabhjot Singh**  
🎓 B.S. in Information Technology, Marymount University  
🔗 [LinkedIn](https://www.linkedin.com/in/prabhjot-singh-10a88b315/)  
🧠 Passionate about data-driven decision making, analytics, and automation

---

## ⭐ Support

If you found this project useful, feel free to ⭐ it and share it with others!
