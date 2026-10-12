#DATA PIPELINE WITH DVC & MLFLOW WITH ML
This project demonstrates how to build an end to end ml pipeline using DVC(Data Version Control) for data & model versioning & MLFLOW for experiment tracking. The pipeline focuses on training a Random Forest Classifier on the Pima Indians Diabetes Dataset, with clear stages for data preprocessing, model training & evaluation.

Key features of the project
DATA VERSION CONTROL(DVC)

DVC is used to track and version the dataset, models and pipeline stages using reproducibility across different environments.The pipeline is structured into stages(preprocessing, training, evaluation) that can be automatically re-executed if any dependencies(data, scripts, parameters) change.

DVC also allows remote data storage(dagshub, S3) for large datasets and models.

Experiment Tracking with MLFlow
MLflow is used to track experiment metrics, parameters & artifacts.It logs the hyperparameters of the model(n_estimators, maxDepth) & performance metrics like accuracy.

MLFlow helps compare different runs and models to optimize the ML pipeline.

Pipeline Stages:

Preprocessing: 
The preprocess.py script reads the raw dataset(data.csv), performs basic preprocessing and outputs the preprocessed data to data.csv
This stage ensures that the data is consistently processed across runs.

Training:
The train.py script trains a Random Forest Classifier on the preprocessed data.
Hyperparameters and the model itself are logged into MLFlow for tracking and comparision.

Evaluation:
The Evaluate.py script loads the trained model and evaluates its performance (accuracy) on the dataset.
The evaluation metrics are logged to MLFlow for tracking.

Scikit-learn : For building and training the Random Forest Classifier.

This project demonstrates how to manage the lifecycle of the ML Project, ensuring the data, code, models and experiments are all traced, versioned and reproducible.

Goals:
Reproducibility - By using DVC,the pipeline ensures that the same data, parameters, and code can reproduce the same results, making the workflow reliable & consistent.

Experimentation - MLFlOW allow users to easily track different experiements(with varying hyperparameters) and compare the performance of models.

FOR ADDING STAGES

1st Stage - Preprocess
dvc stage add -n preprocess
-p preprocess.input,preprocess.output
-d src/preprocess.py -d data/raw/data.csv
-o data/processed/data.csv
python src/preprocess.py

2nd Stage - Train
-p train.data,train.model,train.random_state,train.n_estimators,train.max_depth
-d src/train.py -d data/raw/data.csv
-o models/model.pkl
python src/train.py

3rd Stage - Evaluate
-d src/evaluate.py -d models/model.pkl -d data/raw/data.csv
python src evaluate.py