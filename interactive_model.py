from flask import Flask, request, render_template
import joblib
import numpy as np
import re
from collections import Counter
import string
import pandas as pd

# Load the model and scaler
model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def classify_text():
    prediction = ""
    analysis = {}
    
    if request.method == "POST":
        user_input = request.form["input_text"]
        
        # Analyze the text content
        analysis = preprocess_text(user_input)
        
        # Define the feature order (must match scaler training order)
        feature_order = ["average_sentence_length", "average_word_length", 
                         "comma_frequency", "punctuation_frequency", 
                         "unique_word_count", "zipf_ratio"]
        
        # Create a DataFrame with feature names
        features_df = pd.DataFrame([analysis], columns=feature_order)
        
        # Debug: Check the runtime feature preparation
        print("Runtime feature analysis:", analysis)
        print("Feature DataFrame before scaling:", features_df)
        
        # Scale the features using the trained scaler
        features_scaled = scaler.transform(features_df)
        
        # Use the scaled features for prediction
        raw_prediction = model.predict(features_scaled)[0]  # Predicts the class label directly
        
        # Debug: Check raw prediction and model classes
        print("Raw prediction (label):", raw_prediction)
        print("Model classes:", model.classes_)
        
        if raw_prediction == "AI": #predictions were inverted for some reason
            prediction = "Human"
        else:
            prediction = "AI"

    return render_template("index.html", prediction=prediction, analysis=analysis)

def preprocess_text(file_content):
    """
    Analyzes text content and extracts relevant features for text classification.

    Args:
        file_content (str): The text content to analyze.

    Returns:
        dict: A dictionary of calculated features for the input text.
    """
    # Word and sentence level analysis
    words = re.findall(r'\b\w+\b', file_content)
    sentences = re.split(r'[.!?]', file_content)

    # Calculate features
    word_lengths = [len(word) for word in words]
    average_word_length = sum(word_lengths) / len(word_lengths) if word_lengths else 0
    average_sentence_length = sum(len(sentence.split()) for sentence in sentences if sentence) / len(sentences) if sentences else 0
    unique_word_count = len(set(words))

    # Comma and punctuation frequencies
    comma_count = file_content.count(',')
    punctuation_count = sum(1 for char in file_content if char in string.punctuation)

    # Calculate Zipf's Law adherence (average ratio of word frequencies)
    word_counts = Counter(words)
    zipf_ratios = calculate_zipf_law(word_counts)
    average_zipf_ratio = sum(zipf_ratios) / len(zipf_ratios) if zipf_ratios else 0

    # Return the processed features
    return {
        "average_sentence_length": average_sentence_length,
        "average_word_length": average_word_length,
        "comma_frequency": comma_count,
        "punctuation_frequency": punctuation_count,
        "unique_word_count": unique_word_count,
        "zipf_ratio": average_zipf_ratio,
    }
    
    
# Define a helper function for calculating Zipf's law compliance
def calculate_zipf_law(word_counts):
    """
    Calculates adherence to Zipf's law for a given word frequency distribution.
    
    Args:
        word_counts (dict): A Counter object containing word frequencies.

    Returns:
        list: Ratios indicating adherence to Zipf's law.
    """
    sorted_counts = sorted(word_counts.values(), reverse=True)
    zipf_ratios = [sorted_counts[i] / sorted_counts[0] for i in range(1, len(sorted_counts))]
    return zipf_ratios  # Ratios to measure adherence to Zipf's law

if __name__ == "__main__":
    app.run(debug=True)
