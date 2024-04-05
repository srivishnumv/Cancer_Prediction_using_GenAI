import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load the combined dataset
combined_data = pd.read_csv('C:/Users/Srivishnu/OneDrive/Desktop/Project_Code/Cancer_Prediction_using_GenAI/combined_data.csv')

# Separate features (X) and labels (y)
X = combined_data.drop(columns=['diagnosis'])
y = combined_data['diagnosis']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode the target variable
y_train_encoded = label_encoder.fit_transform(y_train)

# Define the XGBoost model
model = XGBClassifier()

# Train the XGBoost model
model.fit(X_train, y_train_encoded)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
