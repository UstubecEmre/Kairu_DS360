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
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

import json 

#%% 
"""
Hedeflenen: Kategorik verileri sayisala cevir, bunu pipeline ile yap. Eksik verileri kategorik veriler icin
en cok tekrar eden deger yani mod ile, sayisal veriler icin ise medyan ile doldur.
Tarih verilerini datetime formatina cevir, sayisal hale getir.

loan_id gibi kardinal degiskenleri kaldir.

Tarih degiskenlerinden yeni degiskenler turet.
"""

DATA_PATH = r'D:/Datasets/loan_dataset.csv'
TARGET_COL = 'loan_status'
LEAKAGE_COLS = ['paid_off_time','past_due_days']

DROP_COLS = ['Loan_ID',TARGET_COL] + LEAKAGE_COLS
# DROP_COLS = ['loan_id', 'loan_status','paid_off_time', 'past_due_days']

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # proje kökü
PROCESSED_DIR = os.path.join(BASE_DIR, "data/processed")
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
            df[col] = pd.to_datetime(df[col], errors='coerce').astype('int64') // 10**9 # dt.floor('s').astype('int64')

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



                                                        #,stratify= y)
    return X_train, X_test, y_train, y_test, pre
# helper functions (yardımcı fonksiyonlarimiz)
def ensure_processed_dir():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    return PROCESSED_DIR


def get_split_data(test_size = 0.2, random_state = 42):
    loan_df = load_loan_data()
    X, y, pre =split_x_and_y(loan_df)
    X_train, X_test, y_train, y_test = train_test_split(X 
                                                        ,y 
                                                        ,test_size = test_size
                                                        ,random_state = random_state)
#%% define class ratio function (sinif orani fonksiyonunu tanimla)
def calculate_class_ratio(y):
    positive = np.sum(y == 1)
    negative = np.sum(y == 0)
    positive_weight = (positive / negative) if positive > 0 else 1
    return positive, negative, positive_weight

def save_processed_data():
    #save train_original.csv (egitim setinin orijinal halini kaydet)
    ensure_processed_dir()
    X_train, X_test, y_train, y_test, preprocessor = get_split_data()  # preprocessor'ı al
    
    # Pipeline ile sayısala çevir
    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)
    
    # train_original ve test_original CSV
    pd.concat([X_train, y_train.rename('target')], axis=1)\
        .to_csv(os.path.join(PROCESSED_DIR, "train_original.csv"), index=False)
    pd.concat([X_test, y_test.rename('target')], axis=1)\
        .to_csv(os.path.join(PROCESSED_DIR, "test_original.csv"), index=False)
    
    # class_weights.json
    positive, negative, positive_weight = calculate_class_ratio(y_train)
    with open(os.path.join(PROCESSED_DIR, "class_weights.json"), "w") as f:
        json.dump({
            "positive": int(positive),
            "negative": int(negative),
            "positive_weight": float(positive_weight)
        }, f, indent=4)
    
    # SMOTE ve undersample
    smote_X, smote_y = SMOTE(random_state=42).fit_resample(X_train_encoded, y_train)
    undersampled_X, undersampled_y = RandomUnderSampler(random_state=42).fit_resample(X_train_encoded, y_train)
    
    # SMOTE ve undersampled CSV (NumPy array olduğu için header yok)
    np.savetxt(os.path.join(PROCESSED_DIR, "train_smote.csv"), 
               np.hstack([smote_X, smote_y.values.reshape(-1,1)]), delimiter=",")
    np.savetxt(os.path.join(PROCESSED_DIR, "train_undersampled.csv"), 
               np.hstack([undersampled_X, undersampled_y.values.reshape(-1,1)]), delimiter=",")
    
    print("Islenmis veriler basariyla kaydedildi.")

    """ I got an error => hata verdi
    ensure_processed_dir()
    X_train, X_test, y_train, y_test, _ = get_split_data()
    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)
 
    
    train_df = pd.concat([X_train, y_train.rename('target')], axis = 1)
    test_df = pd.concat([X_test, y_test.rename('target')], axis = 1)
    
    train_path = os.path.join(PROCESSED_DIR, "train_original.csv")
    test_path = os.path.join(PROCESSED_DIR, "test_original.csv")
    
    train_df.to_csv(train_path, index = False)
    test_df.to_csv(test_path, index = False)
    
    #save original class weights
    # save class_weights.json
    positive, negative, positive_weight = calculate_class_ratio(y_train)
    
    weights_path = os.path.join(PROCESSED_DIR, "class_weights.json")
    with open(weights_path, mode = "w") as file:
        json.dump({
            "positive":int(positive)
            ,"negative":int(negative)
            ,"positive_weight": float(positive_weight)
        }
            ,file
            ,indent = 4)
    
    # save train_smote.csv
    smote_X, smote_y = SMOTE(random_state=42).fit_resample(X_train_encoded, y_train)
    smote_df = pd.DataFrame(smote_X, columns = X_train.columns)
    smote_df['target'] = smote_y 
    smote_path = os.path.join(PROCESSED_DIR, "train_smote.csv")
    smote_df.to_csv(smote_path, index = False)   
        
        # save undersampled csv
    undersampled_X, undersampled_y = RandomUnderSampler(random_state = 42).fit_resample(X_train_encoded, y_train)
    undersampled_df = pd.DataFrame(undersampled_X, columns = X_train.columns)
    undersampled_df['target'] = undersampled_y
    undersampled_path = os.path.join(PROCESSED_DIR, "train_undersampled.csv")
    undersampled_df.to_csv(undersampled_path, index = False)
    print("Islenmis veriler basariyla kaydediliyor...")
    
    
    # return all train and test files
    return {
        "train_original": "data/processed/train_original.csv",
        "test_original": "data/processed/test_original.csv",
        "train_smote": "data/processed/train_smote.csv",
        "train_undersampled": "data/processed/train_undersampled.csv",
        "class_weights": "data/processed/class_weights.json"
    }
"""
#%% call main function (ana fonksiyonu cagir)
if __name__ == '__main__':
    X_train, X_test, y_train, y_test, preprocessor = get_split_data()
    print('Egitim Setinin Boyutu:', X_train.shape)
    print('Test Setinin Boyutu:', X_test.shape)
    print('Egitim Setindeki Hedef Degiskenin Frekans Dagilimi:\n', y_train.value_counts())
    print('Egitim Setindeki Hedef Degiskenin Oransal Dagilimi:\n', y_train.value_counts(normalize = True).round(4).to_dict())
    save_processed_data()