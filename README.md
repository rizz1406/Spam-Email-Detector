**Spam Email Classifier using Python and Streamlit**

## Project Overview

This project demonstrates a simple **Spam Email Classifier** using Natural Language Processing (NLP) and machine learning techniques. The classifier is capable of predicting whether a given message is **spam** or **ham** (not spam) based on the message content. The model is trained on the popular **SMS Spam Collection Dataset** and is deployed via a **Streamlit** web application for easy user interaction.

## Features

- **Machine Learning Model**: Naive Bayes classifier
- **Text Preprocessing**: TF-IDF (Term Frequency-Inverse Document Frequency) for feature extraction
- **Web Application**: User-friendly interface built with Streamlit for real-time message classification
- **Dataset**: SMS Spam Collection dataset from the UCI Machine Learning Repository
- **Performance**: Achieved ~98% accuracy on the test data

## Technologies Used

- **Python**: Core language for data processing, machine learning, and web app development
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning model and feature extraction
- **Streamlit**: Framework for building a simple web interface
- **NLP**: Natural Language Processing for text analysis and classification
- **TF-IDF**: For converting text to numerical feature vectors

## Installation & Setup

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/your-username/spam-classifier.git
   ```

2. Navigate to the project directory:
   ```bash
   cd spam-classifier
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run spam_classifier.py
   ```

5. Open your browser and go to `http://localhost:8501/`. Enter a message to classify it as **spam** or **ham**.

## Dataset

The dataset used in this project is the **SMS Spam Collection** dataset, which can be downloaded from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection). It consists of 5,574 messages labeled as spam or ham.

## Project Structure

```bash
.
├── spam_classifier.py                 # Main Python script to run the Streamlit app
├── spam.csv                           # SMS Spam Collection dataset (must be in this folder)
├── requirements.txt                   # Dependencies for the project
└── spam_classifier_project.ipynb      # Jupyter notebook with a step-by-step guide
```

## How It Works

1. **Loading the Dataset**: The SMS Spam Collection dataset is loaded and cleaned. The labels (`spam` or `ham`) are mapped to binary values (1 for spam, 0 for ham).
   
2. **Text Vectorization**: The message content is transformed into numerical features using **TF-IDF vectorization**, which converts the raw text into a format that can be used by the machine learning model.
   
3. **Model Training**: A **Naive Bayes classifier** is trained using the vectorized text data. This model is known to perform well on text classification problems like spam detection.
   
4. **Prediction & Evaluation**: The trained model is evaluated on the test data, and metrics such as **accuracy** and **confusion matrix** are computed.

5. **Streamlit Application**: The model is deployed on a Streamlit web app where users can input a message and receive a real-time prediction (spam/ham).

## Example Output

When you input a message into the Streamlit app, it will classify it as **Spam** or **Ham** based on the trained model.

## Future Improvements

- Enhance the UI and add more interactivity to the web app.
- Implement additional machine learning models (e.g., SVM, deep learning models) for improved accuracy.
- Incorporate other datasets to improve the generalization of the classifier.
- Add NLP techniques for better preprocessing, such as lemmatization or stemming.
