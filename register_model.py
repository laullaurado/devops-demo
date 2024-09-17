from comet_ml import API
import os

# Initialize Comet API
api = API(api_key=os.getenv("COMET_API_KEY"))

# Register the model
def register_model(model_name, accuracy):
    workspace = "laullaurado"
    project_name = "devops-demo"
    
    # Register the model in Comet Model Registry
    registered_model = api.register_model(workspace=workspace, model_name=model_name)
    
    # Add a version with metadata
    registered_model.add_version(
        model_path=f"models/{model_name}.pkl",  # Path to your saved model
        metadata={"accuracy": accuracy}
    )

# Compare and register best model
def compare_and_register_best_model():
    # Read accuracies of both models
    with open("logistic_regression_accuracy.txt", "r") as f:
        logistic_regression_accuracy = float(f.read())
    
    with open("random_forest_accuracy.txt", "r") as f:
        random_forest_accuracy = float(f.read())

    # Compare the models
    if logistic_regression_accuracy > random_forest_accuracy:
        register_model("logistic_regression", logistic_regression_accuracy)
    else:
        register_model("random_forest", random_forest_accuracy)

if __name__ == "__main__":
    compare_and_register_best_model()
