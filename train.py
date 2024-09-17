import os
import argparse
import comet_ml
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize the Comet experiment


def init_comet_experiment():
    experiment = comet_ml.Experiment(
        api_key=os.getenv("COMET_API_KEY"),
        project_name="devops-demo",
        workspace="laullaurado",
        log_env_details=False,  # Disable logging of environment details
        log_git_metadata=False,  # Disable git metadata logging
    )
    return experiment

# Plot and save confusion matrix


def plot_confusion_matrix(cm, model_type):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Setosa', 'Versicolor', 'Virginica'],
                yticklabels=['Setosa', 'Versicolor', 'Virginica'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {model_type}')
    plt.savefig(f"{model_type}_confusion_matrix.png")
    plt.close()

# Train a model using Logistic Regression or Random Forest


def train_model(model_type):
    # Load the Iris dataset
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42)

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
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')
    cm = confusion_matrix(y_test, predictions)

    # Plot confusion matrix
    plot_confusion_matrix(cm, model_type)

    return model, accuracy, precision, recall, f1, cm


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help="Model type (logistic_regression or random_forest)")
    args = parser.parse_args()

    # Initialize Comet experiment
    experiment = init_comet_experiment()

    # Train the model and get performance metrics
    model, accuracy, precision, recall, f1, cm = train_model(args.model)

    # Log metrics to Comet
    experiment.log_metric("accuracy", accuracy)
    experiment.log_metric("precision", precision)
    experiment.log_metric("recall", recall)
    experiment.log_metric("f1_score", f1)
    experiment.log_confusion_matrix(cm)

    # Ensure the models/ directory exists
    if not os.path.exists("models"):
        os.makedirs("models")

    # Save the model to disk using joblib
    joblib.dump(model, f"models/{args.model}.pkl")
    experiment.log_model(args.model, f"models/{args.model}.pkl")

    # Save accuracy for comparison in the next step
    with open(f"{args.model}_accuracy.txt", "w") as f:
        f.write(str(accuracy))

    # End the Comet experiment
    experiment.end()
