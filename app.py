from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Load the trained model
model_pipeline = None
try:
    model_pipeline = joblib.load('salary_prediction_model.pkl')
    print("Model 'salary_prediction_model.pkl' loaded successfully!")
except FileNotFoundError:
    print("Error: 'salary_prediction_model.pkl' not found.")
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/predict_salary', methods=['POST'])
def predict_salary():
    if model_pipeline is None:
        return jsonify({"error": "Model not loaded. Cannot make predictions."}), 500

    try:
        data = request.get_json(force=True)
        required_fields = ['Age', 'YearsExperience', 'EducationLevel', 'JobTitle', 'Country']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        input_df = pd.DataFrame([data])
        input_df['Age'] = pd.to_numeric(input_df['Age'], errors='coerce')
        input_df['YearsExperience'] = pd.to_numeric(input_df['YearsExperience'], errors='coerce')

        if input_df['Age'].isnull().any() or input_df['YearsExperience'].isnull().any():
            return jsonify({"error": "Age and YearsExperience must be valid numbers."}), 400

        prediction = model_pipeline.predict(input_df)[0]
        return jsonify({"predicted_salary": float(prediction)})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500

@app.route('/')
def home():
    return jsonify({
        "message": "Salary Prediction API is running!",
        "endpoints": {
            "predict_salary": "POST /predict_salary"
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)