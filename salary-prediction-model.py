import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import xgboost as xgb
import warnings
import joblib

warnings.filterwarnings('ignore')  # Clean output

# --- 1. Load Dataset ---
print("Loading Salary Data...")
df = pd.read_csv('Salary Data.csv')  # Update path if needed

# --- 2. Preprocessing ---
print("Cleaning and preparing data...")

# Rename salary column for consistency
df.rename(columns={'Salary': 'avg_salary',
                   'Years of Experience': 'experience_years',
                   'Education Level': 'education_level'}, inplace=True)

# Handle missing values
df['education_level'].fillna('Unknown', inplace=True)
df['experience_years'].fillna(df['experience_years'].median(), inplace=True)

# Sanity check for numeric types
df['experience_years'] = pd.to_numeric(df['experience_years'], errors='coerce')
df['avg_salary'] = pd.to_numeric(df['avg_salary'], errors='coerce')

# Drop any remaining missing
df.dropna(inplace=True)

# --- 3. Define Features and Target ---
X = df[['age', 'job_simp', 'education_level', 'experience_years']]
y = df['avg_salary']

# Sanity check
print("Salary range:", y.min(), "to", y.max())
print(y.sample(5), "\n")

# Define feature types
categorical_features = ['job_simp', 'education_level']
numerical_features = ['age', 'experience_years']

# --- 4. Preprocessing Pipeline ---
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ]
)

# --- 5. Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# --- 6. Define Models ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
lgbm_model = lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1)
xgb_model = xgb.XGBRegressor(n_estimators=100, objective='reg:squarederror', random_state=42, n_jobs=-1)

# Ensemble
voting_regressor = VotingRegressor([
    ('rf', rf_model),
    ('lgbm', lgbm_model),
    ('xgb', xgb_model)
], n_jobs=-1)

# Final pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', voting_regressor)
])

# --- 7. Train ---
print("\nTraining model...")
pipeline.fit(X_train, y_train)

# --- 8. Evaluate ---
y_pred = pipeline.predict(X_test)
print("\nModel Evaluation:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

# --- 9. Predict Example ---
example = pd.DataFrame([{
    'age': 30,
    'job_simp': 'Data Scientist',
    'education_level': 'Masters',
    'experience_years': 3
}])
predicted_salary = pipeline.predict(example)[0]
print(f"\nPredicted salary for example employee: ${predicted_salary:.2f}")

# --- 10. Save Model ---
joblib.dump(pipeline, 'salary_prediction_model.pkl')
print("\nModel saved as 'salary_prediction_model.pkl'")
