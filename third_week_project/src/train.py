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


