import os
from comet_ml import Experiment

# Define API Key, Workspace, Project
COMET_API_KEY = os.getenv("COMET_API_KEY")
workspace = "laullaurado"
project_name = "devops-demo"

# Initialize experiment
experiment = Experiment(api_key=COMET_API_KEY,
                        workspace=workspace, project_name=project_name)

# Function to register a model


def register_model(model_name, accuracy):
    model_file = f"models/{model_name}.pkl"

    # Log the model to the experiment
    print(f"Logging and registering {model_name} with accuracy {accuracy}")
    experiment.log_model(model_name, model_file)

    # Register the model in the Model Registry
    experiment.register_model(model_name)

# Function to compare and register the best model


def compare_and_register_best_model():
    # Read model metrics
    metrics = {}
    for model_name in ["logistic_regression", "random_forest"]:
        with open(f"{model_name}_accuracy.txt", "r") as f:
            accuracy = float(f.read().strip())
        metrics[model_name] = {'accuracy': accuracy}

        # Optionally add more metrics if saved separately

    # Find the best model
    best_model_name = max(metrics, key=lambda k: metrics[k]['accuracy'])
    best_accuracy = metrics[best_model_name]['accuracy']

    # Register the best model
    register_model(best_model_name, best_accuracy)

    print(f"Best model is {best_model_name} with accuracy {best_accuracy}")


if __name__ == "__main__":
    compare_and_register_best_model()
