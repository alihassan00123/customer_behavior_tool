import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

# 1. Load the customer data
data = pd.read_csv("../data/customer_data.csv")


# 3. Check if 'Churn' column exists
if 'is_churn' not in data.columns:
    raise Exception("🛑 The dataset must contain a 'Churn' column")

# 4. Prepare features and target
X = data[['Recency']].copy()
y = data['is_churn']

# 5. Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Train Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 7. Save model
os.makedirs("../model", exist_ok=True)
joblib.dump(model, "../model/churn_model.pkl")

print("✅ churn_model.pkl saved successfully!")
