#%% import required libraries (Gerekli kutuphaneleri import et)
import os # for operating system related operations (isletim sistemi ile ilgili islemler icin)

import pandas as pd # for data manipulation (veri manipule etme icin)
import numpy as np #for numerical operations (sayisal islemler icin)
import matplotlib.pyplot as plt # for data visualization (veri gorsellestirme icin)
import seaborn as sns # for statistical data visualization (istatistiksel veri gorsellest

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


""" 
Amac: Loan veri seti uzerinde model egitimi yapmak.
Dengesiz veri seti ile bas etmek icin SMOTE ve RandomUnderSampler kullan.
Burada class_weight = 'balanced' parametresinin nasil kullanilabilecegini goster.
XGBoost modelinde scale_pos_weight parametresinin nasil kullanilabilecegini goster.
Sentetik veri olusturmak icin SMOTE kullan.
Ilgili dosyalari kaydet: artifacts isminde bir klasor olustur ve icerisine model, encoder ve
ozellik isimlerini kaydet.
"""


#%% create artifacts directory if not exists (artifacts klasoru yoksa olustur)
def ensure_artifacts_dir():
    artifacts_dir = 'artifacts'
    os.makedirs(artifacts_dir, exist_ok=True)
    return artifacts_dir


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
