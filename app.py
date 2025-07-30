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
        required_fields = ['age', 'hourly', 'employer_provided', 'job_state', 'python_yn', 'job_simp', 'seniority', 'min_salary', 'max_salary', 'num_comp']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Convert input data to DataFrame
        input_df = pd.DataFrame([data])

        # Validate numerical fields
        input_df['age'] = pd.to_numeric(input_df['age'], errors='coerce')
        input_df['hourly'] = pd.to_numeric(input_df['hourly'], errors='coerce')
        input_df['employer_provided'] = pd.to_numeric(input_df['employer_provided'], errors='coerce')
        input_df['python_yn'] = pd.to_numeric(input_df['python_yn'], errors='coerce')
        input_df['min_salary'] = pd.to_numeric(input_df['min_salary'], errors='coerce')
        input_df['max_salary'] = pd.to_numeric(input_df['max_salary'], errors='coerce')
        input_df['num_comp'] = pd.to_numeric(input_df['num_comp'], errors='coerce')


        if input_df[required_fields].isnull().any().any():
         return jsonify({"error": "Some required fields contain invalid or missing values."}), 400

        # Make prediction
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