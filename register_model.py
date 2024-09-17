from comet_ml.api import API, APIExperiment
import os

WORKSPACE = "laullaurado"
PROJECT_NAME = "devops-demo"
# Initialize Comet API
api = API(api_key=os.getenv("COMET_API_KEY"))

experiment = APIExperiment(workspace=WORKSPACE, project_name=PROJECT_NAME)

# Register the model
def register_model(model_name, accuracy):
    
    # Register the model in Comet Model Registry
    registered_model = experiment.register_model(model_name=model_name, version="1.0.0", workspace=WORKSPACE, registry_name="devops-demo")
    
    # Add a version with metadata
    # registered_model.add_version(
    #     model_path=f"models/{model_name}.pkl",  # Path to your saved model
    #     metadata={"accuracy": accuracy}
    # )

# Compare and register best model
def compare_and_register_best_model():
    # Read accuracies of both models
    with open("logistic_regression_accuracy.txt", "r") as f:
        logistic_regression_accuracy = float(f.read().strip())

    with open("random_forest_accuracy.txt", "r") as f:
        random_forest_accuracy = float(f.read().strip())

    # Compare the models
    if logistic_regression_accuracy > random_forest_accuracy:
        register_model("logistic_regression", logistic_regression_accuracy)
    else:
        register_model("random_forest", random_forest_accuracy)

if __name__ == "__main__":
    compare_and_register_best_model()
