from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the saved model and scaler
model = joblib.load('lung_cancer_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    input_data = request.get_json()
    input_data = [[input_data['GENDER'], input_data['AGE'], input_data['SMOKING_STATUS'], 
                   input_data['YELLOW_FINGERS'], input_data['ANXIETY'], input_data['PEER_PRESSURE'], 
                   input_data['CHRONIC_DISEASE'], input_data['FATIGUE'], input_data['ALLERGY'], 
                   input_data['WHEEZING'], input_data['ALCOHOL_CONSUMING'], input_data['COUGHING'], 
                   input_data['SHORTNESS_OF_BREATH']]]
    
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Make a prediction
    prediction = model.predict(input_data_scaled)
    result = 'YES' if prediction[0] == 1 else 'NO'
    
    # Return the result as JSON
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
