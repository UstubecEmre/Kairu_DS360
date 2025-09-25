#%% import required libraries
import os # for operating system operations
import pandas as pd 

from sklearn.model_selection import train_test_split # for train and test split
# for modeling
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 

# for model evaluation
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

# for json operation
import joblib
import json


#%% define train_model() function
# use type annotation:)
def train_model(model_type:str = 'random_forest'):
    """
    Trains a machine learning model and saves its performance metrics.

    This function trains a specified machine learning model on the provided
    training data and evaluates its performance. The evaluation metrics are
    then saved to a JSON file.

    Args:
        model_type (str, optional): The type of model to train.
                                    Accepted values: 'random_forest', 'svc', etc.
                                    Defaults to 'random_forest'.
    """


    # use processed dataset
    iris_processed_df = pd.read_csv('data/processed/iris_processed.csv')
    
    # show first five 
    print("******** First Five Rows ********")
    iris_processed_df.head()
    iris_processed_df.info()
    
    # split X and Y
    try:
        iris_features = ['sepal_length','sepal_width','petal_length','petal_width','sepal_ratio','petal_ratio']
        X = iris_processed_df[iris_features]
        y = iris_processed_df['species_encoded']
        # y = iris_processed_df.drop(iris_features, axis = 1) => DataFrame
        print("X and Y splitted")
    except Exception as err:
        raise Exception(f"An unexpected error occured: {err}")

    # train and test split
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size = 0.2, stratify=y)
    
    
    # model selection
    model = None
    if not isinstance(model_type, str):
        print("Model type must be string.")
        
    elif model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators= 50
            ,max_depth = 10
            ,min_samples_leaf= 5
            ,min_samples_split= 5
            ,random_state= 42 
        ) 
        
    elif model_type == 'svc':
        model = SVC(
            C= 1.0
            ,gamma= 'scale'
            ,kernel='rbf'
            #,probability = True
        )
    elif model_type == 'log_reg':
        model = LogisticRegression(
            penalty='l1'
            ,solver= 'liblinear'
            ,tol = 0.001
            ,C= 1
            ,max_iter= 150
            ,random_state=42
        )    
        
    else:
        raise ValueError(f"Invalid model_type :{model_type} ")   
    
    
    # fit model  
    print(f"\n{model_type} model is training")
    model.fit(X_train, y_train) # set the size of y to one 

   # if you want to use get_dummies(drop_first = True), you can these
   # y = [virginica, species_versicolor, species_setosa] => using get dummies(drop_first = True)
   # y = iris_processed_df.drop(iris_features, axis=1).idxmax(axis=1)
    
    # prediction
    y_pred = model.predict(X_test)
    # y_pred_prob = model.predict_proba(X_test) 
    
    
    # Model evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average= 'weighted')
    
    
    # save models
    os.makedirs(name = 'models', exist_ok= True)
    model_path = f'models/{model_type}_model.pkl'
    joblib.dump(model, model_path) # for saving to harddisk
    
    # save metrics
    metrics = {
        'model_type': model_type
       ,'accuracy_score': float(acc)
       ,'f1_score':float(f1)
       ,'n_features': len(iris_features)
       ,'n_train_samples':len(X_train)
       ,'n_test_sample':len(X_test)
    
    }
    
    
    # save model metrics to JSON format
    with open('models/metrics.json', mode='w') as file:
        json.dump(metrics, file, indent=4)
    
    
    with open('models/features.json',mode = 'w') as file:
        json.dump(iris_features, file, indent= 4)
        
    print(f"Model Trained: {model_type}")
    print(f"Accuracy Score of Model: {acc:.4f}")
    print(f"Model Saved: {model_path}")
    
    
    # more detailed model evaluation
    report = classification_report(y_test, y_pred)
    print(f"Classification Report:\n{report}")
    
    return model, metrics



#%% call the function:
if __name__ == '__main__':
    # random_forest model
    rf_model, rf_metrics = train_model('random_forest')
    
    # support vector classifier model
    svc_model, svc_metrics = train_model('svc')
    
    # logistic regression model
    lr_model, lr_metrics = train_model('log_reg')
    
    ## print("\n******Models are trained and saved:******)")
    print("All models successfully trained and stored in the 'models' directory.")
    