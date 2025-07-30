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
import joblib

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- 1. Load Dataset ---
print("Loading dataset...")
df = pd.read_csv(r'C:\Users\hp\Salary-Prediction-Using-Ensemble-Learning\eda_data.csv')
print("Dataset loaded successfully.")
print(df.head())
print(f"Dataset shape: {df.shape}\n")

# --- 2. Data Preprocessing ---

# Define features (X) and target (y)
X = df[['age', 'hourly', 'employer_provided', 'min_salary', 'max_salary', 'avg_salary', 'job_state', 'python_yn', 'R_yn', 'spark', 'aws', 'excel', 'job_simp', 'seniority', 'desc_len', 'num_comp']]
y = df['avg_salary']  # Use 'avg_salary' as the target variable

# Identify categorical and numerical features
categorical_features = ['job_state', 'job_simp', 'seniority']
numerical_features = ['age', 'hourly', 'employer_provided', 'min_salary', 'max_salary', 'num_comp']

# Create a column transformer for preprocessing
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
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# LightGBM Regressor
lgbm_model = lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# XGBoost Regressor
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror', n_jobs=-1)

# Create pipelines for each individual model
pipeline_rf = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', rf_model)])
pipeline_lgbm = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', lgbm_model)])
pipeline_xgb = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', xgb_model)])

# --- 4. Implement Voting Regressor ---
voting_regressor = VotingRegressor(
    estimators=[
        ('rf', rf_model),
        ('lgbm', lgbm_model),
        ('xgb', xgb_model)
    ],
    n_jobs=-1
)

pipeline_voting = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', voting_regressor)])

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
new_employee_data = pd.DataFrame({
    'age': [30],
    'hourly': [0],
    'employer_provided': [0],
    'min_salary': [50000],
    'max_salary': [100000],
    'avg_salary': [75000],
    'job_state': ['CA'],
    'python_yn': [1],
    'R_yn': [0],
    'spark': [1],
    'aws': [1],
    'excel': [1],
    'job_simp': ['Data Scientist'],
    'seniority': ['Junior'],
    'desc_len': [300],
    'num_comp': [3]
})

predicted_salary = pipeline_voting.predict(new_employee_data)[0]
print(f"Predicted salary for the new employee: ${predicted_salary:.2f}")

# --- Save the Trained Model ---
joblib.dump(pipeline_voting, 'salary_prediction_model.pkl')
print("Trained model saved as 'salary_prediction_model.pkl'")