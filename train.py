import argparse
import comet_ml
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Initialize the Comet experiment
def init_comet_experiment():
    experiment = comet_ml.Experiment(
        api_key=os.getenv("COMET_API_KEY"),  # Fetch API key from environment variable
        project_name="devops-demo",  # Name of the project in Comet
        workspace="laullaurado"  # Your workspace in Comet
    )
    return experiment

# Train a model using Logistic Regression or Random Forest
def train_model(model_type):
    # Load the Iris dataset
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

    # Select the model type
    if model_type == "logistic_regression":
        model = LogisticRegression(max_iter=200)
    elif model_type == "random_forest":
        model = RandomForestClassifier()
    else:
        raise ValueError("Unknown model type")

    # Train the model
    model.fit(X_train, y_train)

    # Predict and evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help="Model type (logistic_regression or random_forest)")
    args = parser.parse_args()

    # Initialize Comet experiment
    experiment = init_comet_experiment()

    # Train the model and get performance metrics
    model, accuracy = train_model(args.model)

    # Log metrics to Comet
    experiment.log_metric("accuracy", accuracy)

    # Save the model to disk using joblib
    joblib.dump(model, f"models/{args.model}.pkl")
    experiment.log_model(args.model, f"models/{args.model}.pkl")

    # Save accuracy for comparison in the next step
    with open(f"{args.model}_accuracy.txt", "w") as f:
        f.write(str(accuracy))

    # End the Comet experiment
    experiment.end()
