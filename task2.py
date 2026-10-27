import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import pickle

# 1. Load Dataset
# Create sample dataset - or replace with StudentsPerformance.csv from Kaggle
data = {
    'Gender': ['Male','Female','Female','Male','Male','Female','Male'],
    'Study_Hours': [2, 5, 6, 3, 8, 4, 7],
    'Attendance': [75, 90, 95, 80, 98, 85, 92],
    'Previous_Score': [60, 75, 80, 65, 90, 70, 85],
    'Internet': ['Yes','Yes','No','Yes','Yes','No','Yes'],
    'Final_Score': [62, 78, 82, 68, 92, 74, 88]
}

df = pd.DataFrame(data)
# If you have CSV: df = pd.read_csv('StudentsPerformance.csv')

# 2. Fill missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# 3. Encode categorical - CORRECTED
cat_cols = ['Gender', 'Internet']
for col in cat_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

# 4. Features and Target
X = df.drop('Final_Score', axis=1)
y = df['Final_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Model - Random Forest Regressor best for this
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluation
y_pred = model.predict(X_test)
print(f"R2 Score: {r2_score(y_test, y_pred)*100:.2f}%")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")

# 7. Save model
with open('student_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nModel saved as student_model.pkl")

# 8. Graph - Very Important for this task
plt.figure()
plt.scatter(df['Study_Hours'], df['Final_Score'], color='blue', label='Actual')
plt.plot(df['Study_Hours'], model.predict(X), color='red', linestyle='--', label='Predicted Trend')
plt.xlabel('Study Hours')
plt.ylabel('Final Score')
plt.title('Student Performance Tracker - Hours vs Score')
plt.legend()
plt.savefig('performance_graph.png')
plt.show()

# 9. Predict function
def predict_performance(gender, hours, attendance, prev_score, internet):
    gender_val = 1 if gender=='Male' else 0
    internet_val = 1 if internet=='Yes' else 0
    input_data = [[gender_val, hours, attendance, prev_score, internet_val]]
    pred = model.predict(input_data)
    return pred[0]

# Example
print(f"\nPredicted Score for 6 hours study: {predict_performance('Male', 6, 90, 75, 'Yes'):.1f}")
