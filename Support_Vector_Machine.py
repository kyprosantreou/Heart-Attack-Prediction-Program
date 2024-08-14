import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Function to train and evaluate a Support Vector Machine (SVM) model
def SVM_train(df):
    # Separate features (X) from the target variable (y)
    X = df.drop('target', axis=1)  # Drop the 'target' column to obtain the feature variables
    y = df["target"]  # Extract the target variable

    # Initialize a standard scaler to normalize the feature data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Scale the feature data to have zero mean and unit variance

    # Split the dataset into training and testing sets (70% training, 30% testing)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.30, random_state=42)

    # Initialize the Support Vector Machine classifier with a linear kernel
    model = svm.SVC(kernel='linear')

    # Train the SVM model using the training data
    model.fit(X_train, y_train)

    # Evaluate the model's accuracy on the test set and print it as a percentage
    print("Accuracy:", model.score(X_test, y_test) * 100)

    # Predict the target values for the test set
    y_pred = model.predict(X_test)

    # Print the classification report to evaluate the model's performance
    print(classification_report(y_test, y_pred))