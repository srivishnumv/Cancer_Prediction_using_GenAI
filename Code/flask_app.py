from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
#C:/Users/Srivishnu/OneDrive/Desktop/Project_Code/Cancer_Prediction_using_GenAI/
# Load the trained XGBoost model
with open('C:/Users/Srivishnu/OneDrive/Desktop/Project_Code/Cancer_Prediction_using_GenAI/Code/xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define a function to preprocess input data
def preprocess_input(input_data):
    # Convert input data to a DataFrame
    input_df = pd.DataFrame([input_data], columns=['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 
                                                   'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean'])
    return input_df


# Define a function to predict cancer
def predict_cancer(input_data):
    # Preprocess the input data
    input_df = preprocess_input(input_data)
    # Make predictions
    predictions = model.predict(input_df)
    # Map prediction results to labels
    prediction_labels = ["Benign" if pred == 0 else "Malignant" for pred in predictions]
    return prediction_labels


# Define route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    radius_mean = float(request.form['radius_mean'])
    texture_mean = float(request.form['texture_mean'])
    perimeter_mean = float(request.form['perimeter_mean'])
    area_mean = float(request.form['area_mean'])
    smoothness_mean = float(request.form['smoothness_mean'])
    compactness_mean = float(request.form['compactness_mean'])
    concavity_mean = float(request.form['concavity_mean'])
    concave_points_mean = float(request.form['concave_points_mean'])

    # Create an array of input values
    input_data = [radius_mean, texture_mean, perimeter_mean, area_mean, 
                  smoothness_mean, compactness_mean, concavity_mean, concave_points_mean]

    # Make predictions
    prediction = predict_cancer(input_data)

    # Render the result template with the prediction
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
