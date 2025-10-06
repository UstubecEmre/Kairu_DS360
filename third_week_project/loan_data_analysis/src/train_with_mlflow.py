#%% import required libraries (gerekli kutuphaneleri import et)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import os
import joblib

from datetime import datetime

import mlflow
import mlflow.sklearn



#%% 
def train_with_ml():
    """MLFlow ile model egitimi gerceklestirir"""
    loan_df = pd.read_csv(r'D:\Datasets\loan_dataset.csv')
    
    for col in ['effective_date','due_date']:
        if col in loan_df.columns:
            loan_df[col] = pd.to_datetime(loan_df[col]
                                          ,errors = "coerce")
    
    if {'Principal', 'terms'}.issubset(loan_df.columns):
        loan_df['principal_per_term'] = np.where(
            loan_df['terms'] == 0
            ,np.nan
            ,loan_df['Principal'] / loan_df['terms']
        )
    
    
    
    # split X and y (Bagimli degiskenleri (X) ve bagimsiz degiskeni (y) ayir)
    feature_cols = ['Principal'
                    ,'terms'
                    ,'age'
                    ,'principal_per_term'
                    ,'education'
                    ,'Gender']
    target_col = 'PAIDOFF'
    
    X = loan_df[feature_cols]
    y = loan_df[target_col]
    
    
    # encoding
    for col in ['education', 'Gender']:
        if col in X.columns:
            X[col] = X[col].astype('category').cat.codes.copy() # encoding
            
    # train and test split
    X_train, X_test, y_train, y_test = train_test_split(X
                                                        ,y
                                                        ,test_size = 0.2
                                                        ,random_state = 42
                                                        , stratify= y)
    
    # mlflow.set_experiment
    mlflow.set_experiment('Loan Paidoff Prediction')
    
    models = {
        'log_reg': LogisticRegression(
            max_iter = 1500
            ,C = 0.8
            ,tol = 0.001
        )
        ,'rf': RandomForestClassifier(
            n_estimators = 150
            ,random_state = 42
            ,max_depth = 10
            ,min_samples_split = 5    
        )
        ,'xgb': XGBClassifier(
             n_estimators = 150
            ,random_state = 42
            ,max_depth = 5
            ,learning_rate = 0.05
            ,subsample = 0.8
            ,colsample_bytree = 0.8
            ,reg_lambda = 1 
            ,eval_metric = "logloss"
        )
    }
    
    # save best auc and accuracy
    best_model_name = None
    best_accuracy = 0
    best_auc = 0
    best_auc_model = None 
    for model_name, model in models.items():
        with mlflow.start_run(run_name=f"{model_name}_experiment"):
            
            # logging 
            if model_name == 'log_reg':
                mlflow.log_param('max_iter',1500)
                mlflow.log_param('C', 0.8)
                mlflow.log_param('tol', 0.001)
        
            elif model_name == 'rf':
                mlflow.log_param('n_estimators', 150)
                mlflow.log_param('random_state', 42)
                mlflow.log_param('max_depth', 10)
                mlflow.log_param('min_samples_split', 5)
            
            elif model_name == 'xgb':
                mlflow.log_param("n_estimators",150)
                mlflow.log_param('random_state', 42)
                mlflow.log_param('max_depth',5)
                mlflow.log_param('learning_rate', 0.05)
                mlflow.log_param('subsample', 0.8)
                mlflow.log_param('colsample_bytree', 0.8)
                mlflow.log_param('reg_lambda', 1)
                mlflow.log_param('eval_metric', 'logloss')
            
            else:
                print("Please, select a valid model name. (rf, log_reg, xgb)")
                # print("Lutfen model olarak Random Forest, XGB veya Logistic Regression seciniz")
                
            mlflow.log_param('model_type', model_name)
            mlflow.log_param('test_size', 0.2)
            mlflow.log_param('n_features', len(feature_cols))
            
            
            # fit models (modelleri egit)
            model.fit(X_train, y_train)
            
            # make a prediction
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            # metrics
            acc = accuracy_score(y_test, y_pred)
            auc_score  = roc_auc_score(y_test, y_proba)
            # save to mlflow
            mlflow.log_metric("accuracy",acc)
            mlflow.log_metric('auc', auc_score)
            mlflow.log_metric('train_size', len(X_train))
            mlflow.log_metric('test_size', len(X_test))
            
            
            # save model
            mlflow.sklearn.log_model(
                sk_model = model
                ,artifact_path = 'model'
                ,registered_model_name = f'loan_credit_{model_name}'
            )
            
            if acc > best_accuracy:
                best_accuracy = acc
                best_model_name = model_name
            print(f"{model_name} - Accuracy: {acc:.4f}")
            if auc_score > best_auc:
                best_auc = auc_score
                best_auc_model = model_name
            print(f"{model_name} - AUC: {auc_score:.4f}")    
    
    print(f"Model With Best Accuracy Score: {best_model_name}\nAccuracy Score: {best_accuracy}")
    #print(f"En Iyi Dogruluk Oranina Sahip Model {best_model_name}\nDogruluk Orani: {best_accuracy:.4f}")
    print(f"Model With Best Area Under Curve Score: {best_auc_model}\nAUC Score: {best_auc}")           
    #print(f"En Iyi Egri Altinda Kalan Alana Sahip Model {best_auc_model}\nEgri Altinda Kalan Alanin Orani: {best_auc:.4f}")
        
    return best_model_name, best_accuracy, best_auc_model, best_auc

if __name__ == '__main__':
    print("MLflow training is starting...")
    # print("MLflow ile model egitimi basliyor...")
    best_acc_model, best_acc, best_auc_model, best_auc = train_with_ml()
    print("If you want to show on MLflow, you can enter: mlflow ui")
    # print("\nMLflow UI'yi Gormek Icin Yapmaniz Gereken: mlflow ui") 
    
            