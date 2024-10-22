# Import necessary libraries
import pandas as pd
import string
import nltk
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords

# Download stopwords if not already present
nltk.download('stopwords')

# Heading: Load and Clean Dataset
def load_data():
    # Load the dataset
    df = pd.read_csv('spam.csv', encoding='latin-1')
    
    # Display column names to help identify them (useful for debugging)
    print(df.columns)

    # Adjust the column names to match your dataset (modify this part based on actual column names)
    df = df[['v1', 'v2']]  # Assuming 'v1' is the label (ham/spam) and 'v2' is the message
    df.columns = ['label', 'message']  # Rename columns for easier access

    # Map labels to 1 for 'spam' and 0 for 'ham'
    df['label'] = df['label'].map({'spam': 1, 'ham': 0})

    return df

# Heading: Preprocess Data (Remove stopwords, punctuation, etc.)
def preprocess_data(df):
    # Removing punctuation and converting to lowercase
    df['message_cleaned'] = df['message'].apply(lambda x: ''.join([char for char in x if char not in string.punctuation]))
    df['message_cleaned'] = df['message_cleaned'].apply(lambda x: x.lower())

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    df['message_cleaned'] = df['message_cleaned'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

    # Vectorize the text data using TF-IDF
    tfidf = TfidfVectorizer(max_features=2500)
    X = tfidf.fit_transform(df['message_cleaned']).toarray()

    # Target variable (label)
    y = df['label']

    return X, y, tfidf

# Heading: Train Model
def train_model(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Multinomial Naive Bayes model
    model = MultinomialNB()

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set and print accuracy and classification report
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))

    return model

# Heading: Run Streamlit App
def run_app(model, tfidf):
    # Set up the Streamlit interface
    st.title('Spam Email Classifier')
    st.write('This app predicts whether a given email is Spam or Not Spam.')

    # Input text box for user to enter an email message
    user_input = st.text_area('Enter your email text here:', height=200)

    if st.button('Classify'):
        if user_input:
            # Preprocess user input and make prediction
            input_cleaned = ''.join([char for char in user_input if char not in string.punctuation])
            input_cleaned = input_cleaned.lower()
            input_cleaned = ' '.join([word for word in input_cleaned.split() if word not in stopwords.words('english')])

            # Convert input text into the same vectorized format as the training data
            input_vectorized = tfidf.transform([input_cleaned]).toarray()

            # Make prediction
            prediction = model.predict(input_vectorized)

            # Output result
            result = 'Spam' if prediction == 1 else 'Not Spam'
            st.write(f'This email is: **{result}**')
        else:
            st.write('Please enter an email to classify.')

# Heading: Main Function
if __name__ == "__main__":
    # Load data
    df = load_data()

    # Preprocess data
    X, y, tfidf = preprocess_data(df)

    # Train the model
    model = train_model(X, y)

    # Run the Streamlit app
    run_app(model, tfidf)
