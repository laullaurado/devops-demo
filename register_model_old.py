
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
    # Read model accuracies
    with open("logistic_regression_accuracy.txt", "r") as f:
        logistic_regression_accuracy = float(f.read().strip())

    with open("random_forest_accuracy.txt", "r") as f:
        random_forest_accuracy = float(f.read().strip())

    # Compare models and register the one with higher accuracy
    if logistic_regression_accuracy > random_forest_accuracy:
        register_model("logistic_regression", logistic_regression_accuracy)
    else:
        register_model("random_forest", random_forest_accuracy)


# Call the comparison and registration function
if __name__ == "__main__":
    compare_and_register_best_model()
