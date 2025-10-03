#%% import required libraries (Gerekli kutuphaneleri import et)
import os # for operating system related operations (isletim sistemi ile ilgili islemler icin)

import pandas as pd # for data manipulation (veri manipule etme icin)
import numpy as np #for numerical operations (sayisal islemler icin)
import matplotlib.pyplot as plt # for data visualization (veri gorsellestirme icin)
import seaborn as sns # for statistical data visualization (istatistiksel veri gorsellest

from preprocessing import get_split_data 

# for model building (model olusturma icin)
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# for resampling (yeniden ornekleme icin)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from preprocessing import preprocess_loan_data

# for ignoring warnings (uyarilari yoksayma icin)
from warnings import filterwarnings
filterwarnings('ignore')

# for saving models and encoders (model ve encoder kaydetme icin)
import joblib
import json

from sklearn.pipeline import Pipeline

""" 
Amac: Loan veri seti uzerinde model egitimi yapmak.
Dengesiz veri seti ile bas etmek icin SMOTE ve RandomUnderSampler kullan.
Burada class_weight = 'balanced' parametresinin nasil kullanilabilecegini goster.
XGBoost modelinde scale_pos_weight parametresinin nasil kullanilabilecegini goster.
Sentetik veri olusturmak icin SMOTE kullan.
Ilgili dosyalari kaydet: artifacts isminde bir klasor olustur ve icerisine model, encoder ve
ozellik isimlerini kaydet.
"""
# ARTIFACTS_DIR = 'artifacts'

#%% create artifacts directory if not exists (artifacts klasoru yoksa olustur)
def ensure_artifacts_dir():
    artifacts_dir = 'artifacts'
    os.makedirs(artifacts_dir, exist_ok=True)
    return artifacts_dir

artifacts_dir = ensure_artifacts_dir()
#%% define class ratio function (sinif orani fonksiyonunu tanimla)
def calculate_class_ratio(y):
    positive = np.sum(y == 1)
    negative = np.sum(y == 0)
    positive_weight = (positive / negative) if positive > 0 else 1
    return positive, negative, positive_weight


#%% evaluate model function (model degerlendirme fonksiyonunu tanimla)
def evaluate_and_print_model(title:str, y_test:np.ndarray, y_pred:np.ndarray, y_proba:np.ndarray):
    if title is None or not isinstance(title, str):
        print("Model basligi gecerli bir string olmalidir.")
        return False
    if y_test is None or y_pred is None or y_proba is None:
        print("y_test, y_pred ve y_proba gecerli degerler olmalidir.")
        return False
    
    print(f"***** {title} *****")
    print("Siniflandirma Raporu (Classification Report):\n",classification_report(y_test, y_pred, digits= 4))
    
    report = classification_report(y_test, y_pred, digits=4, output_dict=True)
    
    try:
        if y_proba.ndim > 1 and y_proba.shape[1] > 1:
            y_proba = y_proba[:, 1] # positive class probabilities (pozitif sinifin olasiligi)
        print("Egri Altindaki Alan (AUC):", roc_auc_score(y_test, y_proba).round(4))
        auc = roc_auc_score(y_test, y_proba).round(4)
    except Exception as err:
        auc = None
        print(f"AUC Hesaplanamadi: {err}")
    
    # for returning values (degerleri dondurmek icin)
    return {
        "classification_report":report,
        "roc_auc":auc
    }


#%% train with different models and resampling techniques (farkli modeller ve yeniden ornekleme teknikleri ile egit)

def scenario_no_resampling(save_tag = '_cw'):
    """Herhangi bir yeniden ornekleme yapmadan model egitimi yapar.
    Kullanilacak makine ogrenme algoritmalari: Logistic Regression, Random Forest, XGBoost"""

    ensure_artifacts_dir()
    X_train, X_test, y_train, y_test, pre = get_split_data()
    
    # save feature names (ozellik isimlerini kaydet)
    with open('{artifacts_dir}/feature_schema.json',"w") as file:
        json.dump({"columns": list(X_train.columns)}, file, indent = 4)
    
    
    # 1- Apply preprocessor =>  Logistic Regression with class_weight = 'balanced'
    log_reg = Pipeline(steps = [
        ("pre",pre),
        ('clf',LogisticRegression(class_weight = 'balanced'
                                  ,max_iter = 15000))
    ]
        
    )
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    y_proba = log_reg.predict_proba(X_test)[:, 1] # positive class probabilities (pozitif sinifin olasiligi)
    
    # evaluate and print model results (model sonuclarini degerlendir ve yazdir)
    log_reg_results = evaluate_and_print_model("Logistic Regression with class_weight = 'balanced'"
                                              ,y_test
                                              ,y_pred
                                              ,y_proba)
    # save model and preprocessor (modeli ve on isleyiciyi kaydet)
    
    joblib.dump(log_reg,
                f"{artifacts_dir}/model_log_reg{save_tag}.pkl")
    
    joblib.dump(pre,
                f"{artifacts_dir}/preprocessor_log_reg{save_tag}.pkl")    
    

# 2 - Apply preprocessor => Random Forest with class_weight = 'balanced'
    # create Random Forest model with class_weight = 'balanced'
    rf_clf = Pipeline(steps = [
        ("pre",pre),
        ('clf',RandomForestClassifier(
            class_weight = "balanced"
            ,n_estimators = 150
            ,random_state = 42
            ,max_depth = 10
            ,min_samples_split = 5))
    ])

    #fit model (modeli egit)
    rf_clf.fit(X_train, y_train)
    y_pred = rf_clf.predict(X_test)
    y_proba = rf_clf.predict_proba(X_test)[:, 1] # positive class probabilities (pozitif sinifin olasiligi)
    # evaluate and print model results (model sonuclarini degerlendir ve yazdir)
    rf_clf_results = evaluate_and_print_model("Random Forest with class_weight = 'balanced'"
                                            ,y_test
                                            ,y_pred
                                            ,y_proba)
    # save model and preprocessor (modeli ve on isleyiciyi kaydet)
    
    joblib.dump(rf_clf,
                f"{artifacts_dir}/model_rf{save_tag}.pkl")
    
    joblib.dump(pre,
                f"{artifacts_dir}/preprocessor_rf{save_tag}.pkl") 
    
    
    # 3 - Apply preprocessor => XGBoost with scale_pos_weight
    _, _, pos_weight = calculate_class_ratio(y_train) # positive, negative, positive_weight
    # create XGBoost model with scale_pos_weight
    xgb = Pipeline(steps = [
        ("pre", pre),
        ("clf", XGBClassifier(
            n_estimators = 500
            ,random_state = 42
            ,max_depth = 7
            ,learning_rate = 0.05
            ,subsample = 0.8
            ,colsample_bytree = 0.8
            ,reg_lambda = 1,
            scale_pos_weight = pos_weight
            ,eval_metric = 'logloss'
        ))
    ])
    
    # fit model (modeli egit)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    y_proba = xgb.predict_proba(X_test)[:, 1] # positive class probabilities (pozitif sinifin olasiligi)
    xgb_results = evaluate_and_print_model("XGBoost with scale_pos_weight"
                                            ,y_test
                                            ,y_pred
                                            ,y_proba)
    
    # save model and preprocessor (modeli ve on isleyiciyi kaydet)
    joblib.dump(xgb,
                f"{artifacts_dir}/model_xgb{save_tag}.pkl")
    joblib.dump(pre,
                f"{artifacts_dir}/preprocessor_xgb{save_tag}.pkl")
    
    return {
        "log_reg": log_reg_results,
        "random_forest": rf_clf_results,
        "xgboost": xgb_results
    }


#%% 