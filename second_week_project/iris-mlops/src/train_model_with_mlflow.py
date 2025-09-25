#%% import required libraries
import os # for operating system operation
import pandas as pd # for data reading, data manipulation
import numpy as np # for array operations
import matplotlib.pyplot as plt # for data visualization
 
# for train and test split and model evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# to build model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# for mlflow
import mlflow 
import mlflow.sklearn

import joblib


#%%
def train_model_with_mlflow():
    """_summary_
    """
    try:
        iris_df = pd.read_csv(r'D:\Kairu_DS360_Projects\data\processed\iris_processed.csv')
        print("First Five Rows")
    except FileNotFoundError:
        raise FileNotFoundError(f"File path not found:")
    
    # select independent variables (X) 
    # select target column (y)
    
    iris_features = ['sepal_length','sepal_width','petal_length','petal_width','sepal_ratio','petal_ratio']
    try:
        X = iris_df[iris_features]
        y = iris_df['species_encoded']
        print("X and y splitted for modelling")
    except Exception as err:
        raise Exception(f"An unexpected error occured: {err}")
    
    #train and test split for modelling

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state= 42, stratify = y)
    
    
    # use MLFlow and give a title
    mlflow.set_experiment('Iris Species Prediction')
    # modelling
    
    models = {
        'logistic_regression':LogisticRegression(
             penalty='l1'
            ,solver= 'liblinear'
            ,tol = 0.001
            ,C= 1
            ,max_iter= 150
            ,random_state=42
        ),
        
        'random_forest': RandomForestClassifier(
            n_estimators= 50
            ,max_depth = 10
            ,min_samples_leaf= 5
            ,min_samples_split= 5
            ,random_state= 42
            ),
        
        'svc':SVC(
            C= 1.0
            ,gamma= 'scale'
            ,kernel='rbf'
            
        )
    }
    
    
    # define best model % best accuracy score
    best_model = None
    best_accuracy_score = 0.0
    best_f1_score = 0.0 
    best_model_f1 = None
    
    for model_name, model in models.items():
        with mlflow.start_run(run_name=f'{model_name}_experiment'):
            # logging
            print(f'{model_name} model is training:)')
            if model_name == 'logistic_regression':
                mlflow.log_param('penalty', 'l1')
                mlflow.log_param('solver', 'liblinear')
                mlflow.log_param('tol', 0.001)
                mlflow.log_param('C', 1)
                mlflow.log_param('max_iter', 150)
            
            elif model_name == 'random_forest':
                mlflow.log_param('n_estimator',50)
                mlflow.log_param('max_depth',10)
                mlflow.log_param('min_samples_leaf', 5)
                mlflow.log_param('min_samples_split', 5)
                
            elif model_name == 'svc':
                mlflow.log_param('C',1)
                mlflow.log_param('gamma', 'scale')
                mlflow.log_param('kernel','rbf')
                mlflow.log_param('probability', True)
            else:
                print('Invalid model name. Please choose valid model name (random_forest, logistic_regression, svc)')

            # Shared parameter logging
            mlflow.log_param('model_type', model_name)
            mlflow.log_param('test_size', 0.2)
            mlflow.log_param('n_features', len(iris_features))
            
            
            # fit model
            model.fit(X_train, y_train)
            
            # make some predictions
            y_pred = model.predict(X_test)
            #y_pred_proba = model.predict_proba(X_test)[:,1]    

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average= 'weighted')
            
            
            # log accuracy score and f1 score
            mlflow.log_metric('accuracy_score', acc)
            mlflow.log_metric('f1_score', f1)
            
            # save the model
            mlflow.sklearn.log_model(
                sk_model = model
               ,artifact_path = 'model'
               ,registered_model_name= f'iris_{model_name}'
            )
            
            
            # Find best model and accuracy
            if best_accuracy_score < acc:
                best_accuracy_score = acc
                best_model = model_name
            
            if best_f1_score < f1:
                best_f1_score = f1
                best_model_f1 = model_name
                
            print(f"{model_name}'s Model Accuracy Score: {acc:.4f}")
            print(f"{model_name}'s Model F1 Score: {f1:.4f}")         
        print(f"Selected Best Model: {best_model}\nAccuracy Score of Best Model: {best_accuracy_score:.4f}")
        print(f"Selected Best Model: {best_model_f1}\nF1 Score of Best Model: {best_f1_score}")
        # return best_model and best_accuracy_score 
        return best_model, best_accuracy_score
    
#%% call the function
if __name__ == '__main__':
    print("****** Training model with MLFlow ******")
    best_model, best_acc = train_model_with_mlflow()
    print('\nTo view MLFLOW UI: mlflow ui')