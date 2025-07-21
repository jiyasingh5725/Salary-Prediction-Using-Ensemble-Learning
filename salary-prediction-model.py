import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

print("Generating synthetic dataset...")
# --- 1. Generate a Synthetic Dataset for Demonstration ---
# In a real scenario, you would load your Kaggle dataset here:
# df = pd.read_csv('your_kaggle_salary_dataset.csv')

np.random.seed(42) # for reproducibility

num_samples = 1000

# Features
ages = np.random.randint(22, 60, num_samples)
years_experience = np.random.uniform(0, 35, num_samples)
education_levels = np.random.choice(['High School', 'Bachelor\'s', 'Master\'s', 'PhD'], num_samples, p=[0.15, 0.45, 0.30, 0.10])
job_titles = np.random.choice(['Software Engineer', 'Data Scientist', 'Project Manager', 'HR Specialist', 'Sales Manager', 'Marketing Analyst'], num_samples, p=[0.3, 0.2, 0.15, 0.1, 0.1, 0.15])
countries = np.random.choice(['USA', 'Canada', 'UK', 'Germany', 'Australia'], num_samples, p=[0.4, 0.2, 0.15, 0.15, 0.1])

# Base salary calculation (simplified)
base_salary = 40000 + (years_experience * 3000) + (ages * 500)

# Add variations based on education, job title, and country
salary_modifiers = {
    'High School': -10000,
    'Bachelor\'s': 0,
    'Master\'s': 15000,
    'PhD': 30000,
    'Software Engineer': 20000,
    'Data Scientist': 25000,
    'Project Manager': 10000,
    'HR Specialist': -5000,
    'Sales Manager': 8000,
    'Marketing Analyst': 5000,
    'USA': 20000,
    'Canada': 10000,
    'UK': 5000,
    'Germany': 7000,
    'Australia': 12000
}

salaries = []
for i in range(num_samples):
    s = base_salary[i]
    s += salary_modifiers[education_levels[i]]
    s += salary_modifiers[job_titles[i]]
    s += salary_modifiers[countries[i]]
    # Add some random noise
    s += np.random.normal(0, 10000)
    salaries.append(max(25000, s)) # Ensure minimum salary

df = pd.DataFrame({
    'Age': ages,
    'YearsExperience': years_experience,
    'EducationLevel': education_levels,
    'JobTitle': job_titles,
    'Country': countries,
    'Salary': salaries
})

print("Synthetic dataset created successfully.")
print(df.head())
print(f"Dataset shape: {df.shape}\n")

# --- 2. Data Preprocessing ---

# Define features (X) and target (y)
X = df.drop('Salary', axis=1)
y = df['Salary']

# Identify categorical and numerical features
categorical_features = ['EducationLevel', 'JobTitle', 'Country']
numerical_features = ['Age', 'YearsExperience']

# Create a column transformer for preprocessing
# Numerical features will be scaled
# Categorical features will be one-hot encoded
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data split into training and testing sets.")
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}\n")

# --- 3. Define Ensemble Models ---

# Random Forest Regressor
# A bagging ensemble method that builds multiple decision trees and merges their predictions.
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# LightGBM Regressor
# A gradient boosting framework that uses tree-based learning algorithms. Known for speed and efficiency.
lgbm_model = lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# XGBoost Regressor
# An optimized distributed gradient boosting library designed to be highly efficient, flexible and portable.
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror', n_jobs=-1)

# Create pipelines for each individual model, combining preprocessing and the regressor
pipeline_rf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('regressor', rf_model)])

pipeline_lgbm = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', lgbm_model)])

pipeline_xgb = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', xgb_model)])

# --- 4. Implement Voting Regressor (Ensemble of Ensembles) ---
# The Voting Regressor will average the predictions of the base models.
# You can assign weights to each base model if desired (e.g., based on their individual performance).
voting_regressor = VotingRegressor(
    estimators=[
        ('rf', rf_model),
        ('lgbm', lgbm_model),
        ('xgb', xgb_model)
    ],
    n_jobs=-1 # Use all available CPU cores for parallel processing
)

# Create a pipeline for the Voting Regressor
pipeline_voting = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('regressor', voting_regressor)])

# --- 5. Train and Evaluate Models ---

models = {
    "Random Forest Regressor": pipeline_rf,
    "LightGBM Regressor": pipeline_lgbm,
    "XGBoost Regressor": pipeline_xgb,
    "Voting Regressor (RF + LGBM + XGBoost)": pipeline_voting
}

print("Training and evaluating models...\n")

for name, model in models.items():
    print(f"--- Training {name} ---")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"{name} Performance:")
    print(f"  Mean Absolute Error (MAE): ${mae:.2f}")
    print(f"  Root Mean Squared Error (RMSE): ${rmse:.2f}")
    print(f"  R-squared (R2): {r2:.4f}\n")

print("Model training and evaluation complete.")

# --- Example of making a prediction with the best performing model (Voting Regressor) ---
print("--- Example Prediction ---")
# Create a new data point for prediction
new_employee_data = pd.DataFrame({
    'Age': [30],
    'YearsExperience': [5],
    'EducationLevel': ['Master\'s'],
    'JobTitle': ['Data Scientist'],
    'Country': ['USA']
})

# Predict salary using the trained Voting Regressor
predicted_salary = pipeline_voting.predict(new_employee_data)[0]
print(f"Predicted salary for the new employee: ${predicted_salary:.2f}")

# You can also access individual base models from the voting regressor if needed
# For example, to see the predictions from individual models within the VotingRegressor:
# base_predictions = []
# for name, estimator in pipeline_voting.named_steps['regressor'].estimators_:
#     # Need to apply preprocessor to new_employee_data first for base estimators
#     processed_data = pipeline_voting.named_steps['preprocessor'].transform(new_employee_data)
#     base_pred = estimator.predict(processed_data)[0]
#     base_predictions.append((name, base_pred))
# print("\nIndividual base model predictions for new employee:")
# for name, pred in base_predictions:
#     print(f"  {name}: ${pred:.2f}")
import joblib

# Assuming 'pipeline_voting' is the name of your trained Voting Regressor pipeline
# This will save the entire pipeline, including preprocessing steps.
joblib.dump(pipeline_voting, 'salary_prediction_model.pkl')
print("Trained model saved as 'salary_prediction_model.pkl'")