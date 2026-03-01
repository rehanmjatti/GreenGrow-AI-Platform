import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load the data - Updated to match YOUR file name exactly
try:
    df = pd.read_csv('Crop Recommendation dataset.csv')
    print("Found the dataset! Starting training...")
except FileNotFoundError:
    print("Error: Could not find 'Crop Recommendation dataset.csv'. Please check the file name.")
    exit()

# 2. Separate Features (Input) and Target (Output)
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# 3. Split data (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 5. Check Accuracy
accuracy = model.score(X_test, y_test)
print(f"Model Training Complete! Accuracy: {accuracy * 100:.2f}%")

# 6. Save the model
joblib.dump(model, 'crop_model.pkl')
print("SUCCESS: 'crop_model.pkl' has been created in your folder.")


