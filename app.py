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
    model_pipeline = joblib.load('salary_model.pkl')
    print(" Model 'salary_prediction_model.pkl' loaded successfully!")

except FileNotFoundError as e:
    print(f"❌ File not found: {e}")
except Exception as e:
    print(f"❌ Error loading model: {e}")


@app.route('/predict_salary', methods=['POST'])
def predict_salary():
    if model_pipeline is None:
        return jsonify({"error": "Model not loaded. Cannot make predictions."}), 500

    try:
        data = request.get_json(force=True)

        # Expected fields from Salary Data.csv
        expected_columns = ['age', 'job_simp', 'education_level', 'experience_years']

        # Check for missing fields
        missing_fields = [col for col in expected_columns if col not in data]
        if missing_fields:
            return jsonify({"error": f"Missing required field(s): {', '.join(missing_fields)}"}), 400

        # Create input DataFrame
        input_df = pd.DataFrame([{
            'age': data['age'],
            'job_simp': data['job_simp'],
            'education_level': data['education_level'],
            'experience_years': data['experience_years']
        }])

        # Convert numeric fields
        input_df['age'] = pd.to_numeric(input_df['age'], errors='coerce')
        input_df['experience_years'] = pd.to_numeric(input_df['experience_years'], errors='coerce')

        if input_df.isnull().any().any():
            return jsonify({"error": "Some input fields are invalid or missing values."}), 400

        # Make prediction
        predicted_salary = model_pipeline.predict(input_df)[0]
        return jsonify({"predicted_salary": round(float(predicted_salary), 2)})

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
