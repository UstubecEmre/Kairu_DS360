#%% import required libraris (Gerekli kutuphaneleri import et)
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# for modelling (modelleme icin)
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


#%% 
"""
Hedeflenen: Kategorik verileri sayisala cevir, bunu pipeline ile yap. Eksik verileri kategorik veriler icin
en cok tekrar eden deger yani mod ile, sayisal veriler icin ise medyan ile doldur.
Tarih verilerini datetime formatina cevir, sayisal hale getir.

loan_id gibi kardinal degiskenleri kaldir.

Tarih degiskenlerinden yeni degiskenler turet.
"""

DATA_PATH = r'D:\Datasets\loan_dataset.csv'
TARGET_COL = 'loan_status'
LEAKAGE_COLS = ['paid_off_time','past_due_days']

DROP_COLS = ['loan_id',TARGET_COL] + LEAKAGE_COLS
# DROP_COLS = ['loan_id', 'loan_status','paid_off_time', 'past_due_days']


# %% define feature engineering function (ozellik muhendisligi fonksiyonunu tanimla)
def _feature_engineering_on_loan_df(df: pd.DataFrame) -> pd.DataFrame:

    # tarih sutunlarini datetime formatina cevir
    date_cols = ['effective_date','due_date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col],errors ='coerce')
            
    # Apply Feature extraction (Yeni ozellik turet)
    if {'effective_date','due_date'}.issubset(df.columns):
        df['planned_term_days'] = (df['due_date']- df['effective_date']).dt.days
        
        
    # convert to timestamp (tarihi sayisala cevir)
    for col in date_cols:
        if col in df.columns:
            # df[col] = pd.to_datetime(df[col],errors = 'coerce').astype(np.int64) // 10 ** 9
            # more readable way (Daha okunabilir yontem)
            df[col] = pd.to_datetime(df[col], errors='coerce').dt.floor('s').astype(int)

    # if 'Principal' and 'terms' in df.columns:
    if 'Principal' in df.columns and 'terms' in df.columns:
        #prevent division by zero error (sifira bolum hatasini engelle)
        terms = df['terms'].replace({0: np.nan})    
        df['principal_per_term'] = df['Principal'] / terms
    
    # return df (df'yi dondur)
    return df 

def load_loan_data()-> pd.DataFrame:
    path = DATA_PATH
    try:
        df = pd.read_csv(path)
        print("Veri Seti Basariyla Yuklendi.")
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset bulunamadi.Lutfen dosya yolunu kontrol edin.{path}")
    
    return df


def split_x_and_y(df: pd.DataFrame):
    if TARGET_COL not in df.columns:
        raise ValueError(f"Hedef degisken '{TARGET_COL}' veri setinde bulunamadi.")
    
    y = df[TARGET_COL].replace(
        {
            'PAIDOFF':0,
            'COLLECTION':1,
            'COLLECTION_PAIDOFF':1
        }
    ).astype(int)
    
    X = df.drop(columns = DROP_COLS, errors= 'ignore').copy()
    X = _feature_engineering_on_loan_df(X)
    
    
    # select categorical and numerical columns (kategorik ve sayisal sutunlari sec)
    num_cols = X.select_dtypes(include = [np.number]).columns.tolist()
    
    # cat_cols = X.select_dtypes(include = 'object').columns.tolist()
    cat_cols = [col for col in X.columns if col not in num_cols 
                and X[col].nunique() < 20] # kardinal degiskenleri kaldir
    
    
    # create preprocessing pipelines for both numeric and categorical data (sayisal ve kategorik veriler icin on isleme pipeline'lari olustur)
    numeric = Pipeline(steps = [
        ('imputer',SimpleImputer(strategy = 'median'))
    ])
    
    categorical = Pipeline(steps = [
        ('imputer',SimpleImputer(strategy = 'most_frequent')),
        ('onehot',OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # create the column transformer (sutun donusturucu olustur)
    preprocessor = ColumnTransformer(
        [
        ("num", numeric, num_cols),
        ('cat', categorical, cat_cols)
        ]
    )
    
    return X, y, preprocessor