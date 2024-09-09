from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the saved model and scaler
model = joblib.load('lung_cancer_model.pkl')
scaler = joblib.load('scaler .pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    input_data = request.get_json()
    
    # Ensure that all 15 features are present in the input
    input_data_list = [
        input_data.get('GENDER', 0), 
        input_data.get('AGE', 0), 
        input_data.get('SMOKING_STATUS', 0), 
        input_data.get('YELLOW_FINGERS', 0), 
        input_data.get('ANXIETY', 0), 
        input_data.get('PEER_PRESSURE', 0), 
        input_data.get('CHRONIC_DISEASE', 0), 
        input_data.get('FATIGUE', 0), 
        input_data.get('ALLERGY', 0), 
        input_data.get('WHEEZING', 0), 
        input_data.get('ALCOHOL_CONSUMING',0), 
        input_data.get('COUGHING',0), 
        input_data.get('SHORTNESS_OF_BREATH',0),
        # Placeholder for additional features if needed
        2,  # For example, 14th feature
        2   # For example, 15th feature
    ]
    
    # Scale the input data
    input_data_scaled = scaler.transform([input_data_list])
    
    # Make a prediction
    prediction = model.predict(input_data_scaled)
    result = 'YES' if prediction[0] == 1 else 'NO'
    
    # Return the result as JSON
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
