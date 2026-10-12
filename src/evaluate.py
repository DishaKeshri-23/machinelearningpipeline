import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import yaml
import os
import mlflow
from urllib.parse import urlparse

os.environ['MLFLOW_TRACKING_URL'] = "https://dagshub.com/dishakeshri23/machinelearningpipeline.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "dishakeshri23"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "mindflayer"

#Load parameters from param.yaml
params = yaml.safe_load(open("params.yaml"))["train"]

def evaluate(data_path,model_path):

    columns = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
    ]

    data = pd.read_csv(data_path, header=None, names=columns)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]


    mlflow.set_tracking_uri("https://dagshub.com/dishakeshri23/machinelearningpipeline.mlflow")  

    #load the model from disk
    model = pickle.load(open(model_path,'rb'))

    predictions = model.predict(X)
    accuracy = accuracy_score(y,predictions)

    #log metrics to MLFLOW
    mlflow.log_metric("accuracy",accuracy)
    print(f"Model accuracy:{accuracy}")


if __name__ == "__main__":                   #This code means run the evaluate() function only when the Python file is executed directly
    evaluate(params["data_path"],params["model_path"])   #If the condition is true, Python executes the indented code below it. Otherwise, it skips that code.


