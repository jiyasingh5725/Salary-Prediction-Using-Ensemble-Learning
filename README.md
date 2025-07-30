# Salary-Prediction-Using-Ensemble-Learning
# Salary Predictor

## Overview

Welcome to the Salary Predictor project! This application provides a data-driven estimation of an individual's potential salary based on key professional attributes like age, job role, education level, and years of experience. Leveraging an ensemble of powerful machine learning models, this tool aims to offer valuable insights into market salary trends.

The project is structured into three main components:

1.  **Machine Learning Model:** A robust pipeline that preprocesses data and uses an ensemble (Random Forest, LightGBM, XGBoost) to predict salaries.
2.  **Flask API:** A backend service that exposes the trained model via a RESTful API, allowing external applications to request salary predictions.
3.  **Web Interface:** A user-friendly frontend built with HTML and Tailwind CSS, enabling users to input their details and receive instant salary predictions.

## Features

* **Accurate Salary Prediction:** Utilizes a highly effective ensemble model for precise salary estimations.
* **Data Preprocessing Pipeline:** Handles missing values, scales numerical features, and encodes categorical features automatically.
* **RESTful API:** Provides a clean and easy-to-use API endpoint for programmatically accessing predictions.
* **Intuitive Web Interface:** A modern and responsive web application for a seamless user experience.
* **Parallax Background Effects:** Enhances the visual appeal of the web interface with dynamic background elements.
* **Model Persistence:** The trained machine learning model is saved for easy deployment and reuse.

## Project Structure
You're building a comprehensive Salary Prediction project that includes a machine learning model, a Flask API, and a web-based user interface. This is a great setup!

Here's a detailed README.md description for your project, covering all the essential aspects:

Markdown

# Salary Predictor

## Overview

Welcome to the Salary Predictor project! This application provides a data-driven estimation of an individual's potential salary based on key professional attributes like age, job role, education level, and years of experience. Leveraging an ensemble of powerful machine learning models, this tool aims to offer valuable insights into market salary trends.

The project is structured into three main components:

1.  **Machine Learning Model:** A robust pipeline that preprocesses data and uses an ensemble (Random Forest, LightGBM, XGBoost) to predict salaries.
2.  **Flask API:** A backend service that exposes the trained model via a RESTful API, allowing external applications to request salary predictions.
3.  **Web Interface:** A user-friendly frontend built with HTML and Tailwind CSS, enabling users to input their details and receive instant salary predictions.

## Features

* **Accurate Salary Prediction:** Utilizes a highly effective ensemble model for precise salary estimations.
* **Data Preprocessing Pipeline:** Handles missing values, scales numerical features, and encodes categorical features automatically.
* **RESTful API:** Provides a clean and easy-to-use API endpoint for programmatically accessing predictions.
* **Intuitive Web Interface:** A modern and responsive web application for a seamless user experience.
* **Parallax Background Effects:** Enhances the visual appeal of the web interface with dynamic background elements.
* **Model Persistence:** The trained machine learning model is saved for easy deployment and reuse.


## Setup and Installation

Follow these steps to set up and run the project locally.

### 1. Clone the Repository

First, clone the project repository to your local machine.

### 2. Create a Virtual Environment (Recommended)

It's highly recommended to use a virtual environment to manage dependencies:

### 3. Install Dependencies

Install all the required Python libraries:

### 4. Prepare the Dataset

Ensure you have your salary data ready. The script expects a CSV file named Salary Data.csv in the root directory of the project. Make sure it contains columns like 'Salary', 'Years of Experience', 'Education Level', 'Age', and 'Job Title'. The app.py script will handle the renaming of columns to 'avg_salary', 'experience_years', and 'education_level' as part of its preprocessing.

### 5. Run the Application

The app.py script performs both the model training/saving and starts the Flask API.

Upon successful execution, you will see output indicating the model training progress, evaluation metrics, and that the Flask application is running.

## Usage

### Web Interface

Once the Flask app is running, open your web browser and navigate to:

**Home Page:**  http://localhost:5000/ (This route serves a JSON message; you might need to manually navigate to http://localhost:5000/homePage.html for the actual home page.)

**Prediction Form:**  http://localhost:5000/index.html

Input the required details (Age, Job Title, Education Level, Years of Experience) into the form and click the "Predict Salary" button to get an estimated salary.

### API Endpoint

You can also interact with the prediction model directly via the API. Send a POST request to http://localhost:5000/predict_salary with a JSON body.

Endpoint: POST /predict_salary

Request Body Example:

Response Body Example (Success):

Response Body Example (Error - Missing Fields):

Response Body Example (Error - Invalid Input):

### Model Details

The machine learning model is an Ensemble Regressor combining:

**RandomForestRegressor**

**LGBMRegressor (Light Gradient Boosting Machine)**

**XGBRegressor (Extreme Gradient Boosting)**

This ensemble approach typically provides more robust and accurate predictions than a single model.

### Preprocessing Steps:

Column Renaming: 'Salary' -> 'avg_salary', 'Years of Experience' -> 'experience_years', 'Education Level' -> 'education_level'.

Missing Value Imputation:

'education_level': Filled with 'Unknown'.

'experience_years': Filled with the median of the 'experience_years' column.

Numeric Conversion: 'experience_years' and 'avg_salary' are converted to numeric types, with coercion for errors.

Categorical Encoding: 'job_simp' and 'education_level' are one-hot encoded.

Numerical Scaling: 'age' and 'experience_years' are standardized using StandardScaler.

The entire preprocessing and modeling steps are encapsulated within a Pipeline for consistent transformations during training and prediction.

# Technologies Used

Python 3.x

pandas

numpy

scikit-learn

LightGBM

XGBoost

Flask

Flask-CORS

joblib (for model persistence)

HTML5

Tailwind CSS

JavaScript

## Future Enhancements

Implement more sophisticated hyperparameter tuning for the ensemble models.

Add more features to the dataset (e.g., location, company size, industry).

Create a user authentication system for personalized predictions.

Improve error handling and validation on both frontend and backend.

Deploy the application to a cloud platform (e.g., AWS, Heroku).





