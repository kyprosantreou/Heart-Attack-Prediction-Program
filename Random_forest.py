import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Function to train and evaluate a Random Forest model
def Random_forest_train(df):
    # Separate features (X) from the target variable (y)
    X = df.drop('target', axis=1)  # Drop the 'target' column to get feature variables
    y = df["target"]  # Target variable is the 'target' column

    # Initialize a standard scaler to normalize the feature data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Scale the feature data to have zero mean and unit variance

    # Split the dataset into training and testing sets (70% training, 30% testing)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.30, random_state=42)

    # Initialize the Random Forest classifier with 100 trees (estimators)
    model = RandomForestClassifier(n_estimators=100)

    # Train the model using the training data
    model.fit(X_train, y_train)

    # Evaluate the model's accuracy on the test set and print it as a percentage
    print("Accuracy:", model.score(X_test, y_test) * 100)

    # Predict the target values for the test set
    y_pred = model.predict(X_test)

    # Print the classification report to evaluate the model's performance
    print(classification_report(y_test, y_pred))