import pandas as pd
import os
from sklearn.cluster import KMeans
import joblib

# Load the dataset (must contain columns: Recency, TransactionCount, TotalSpent)
data = pd.read_csv("../data/customer_data.csv")



# Use the given columns to create RFM features
data['Frequency'] = data['TransactionCount']  # Rename to match RFM format
data['Monetary'] = data['TotalSpent']         # Rename to match RFM format

# Select the features for training
X = data[['Recency', 'Frequency', 'Monetary']]

# Train KMeans model
model = KMeans(n_clusters=4, random_state=42)
model.fit(X)

# Save the trained model to the 'model' folder
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/cluster_model.pkl")

print("✅ Model trained and saved to model/cluster_model.pkl")
