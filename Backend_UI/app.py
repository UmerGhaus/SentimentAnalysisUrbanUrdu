from flask import Flask, render_template, request
import joblib
import pandas as pd
import re
import string

# Initialize the Flask application
app = Flask(__name__)

# Load the pre-trained model and vectorizer
model = joblib.load('model.pkl')  # Replace with your model's filename
vectorizer = joblib.load('vectorizer.pkl')      # Replace with your vectorizer's filename

# Preprocessing function
def preprocess_text(text):
    # Load stopwords
    stopwords = set(pd.read_csv('RomanUrduStopwords.csv', header=None)[0])
    # Lowercase the text
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(f"[{string.punctuation}0-9]", " ", text)
    # Remove stopwords
    text = " ".join(word for word in text.split() if word not in stopwords)
    return text

@app.route('/')
def home():
    return render_template('index.html')  # Render the homepage with the form

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the text input from the form
        review = request.form['review']
        
        # Preprocess and vectorize the input text
        clean_review = preprocess_text(review)
        review_vect = vectorizer.transform([clean_review])
        
        # Predict the sentiment
        prediction = model.predict(review_vect)
        
        # Map prediction to label
        if prediction[0] == 'pos':  # Replace 'pos' and 'neg' with your model's actual output labels
            result = 'Positive'
        elif prediction[0] == 'Neutral':
            result = 'Neutral'
        else:
            result = 'Negative'
        
        # Return the result along with the original review text
        return render_template('index.html', prediction_text=f'Sentiment: {result}', review_text=review)


if __name__ == "__main__":
    app.run(debug=True)
