# import required libraries (Onemli kutuphaneleri import et)
import os
import numpy as np
import pandas as pd
from pathlib import Path
# import kagglehub 

try:
    PROJECT_DIR = Path(__file__).resolve().parents[1] # fraud detection klasoru altinda olussun, her cagrildiginda ilgili klasorde olusmasin.
except NameError:
    PROJECT_DIR = Path.cwd()
    
    
DATA_DIR = PROJECT_DIR / "data" / "raw"

def ensure_raw_data():
    if not DATA_DIR.exists():
        os.makedirs(DATA_DIR, exist_ok= True)
        print(f"{DATA_DIR} Klasoru Olusturuldu")
    else:
        print(f"{DATA_DIR} Klasoru Mevcut")


def load_dataset(dataset_path:str):
    ensure_raw_data()
    if not isinstance(dataset_path,str) or not dataset_path:
        #print("Veri Setinizin Yolu String Formatta Olmalı ve Boş Olmamalıdır")
        #return False
        raise ValueError("Veri Seti Yolu String Girilmeli ve Bos Birakilmamalidir")
    try:
       # dataset_path = r"D:/Datasets/creditcard.csv"
        fraud_df = pd.read_csv(dataset_path)
        print("Fraud DataFrame'i Basarili Bir Sekilde Yuklendi.")
    except FileNotFoundError:
        raise FileNotFoundError(f"Dosya Yolu Bulunamadi : {dataset_path}! Lutfen Dogru Bir Dosya Yolu Giriniz")
    except Exception as err:
        raise Exception(f"Beklenmeyen Bir Hata Gerceklesti: {err}")
    
    target_path = Path(DATA_DIR) / "creditcard_fraud.csv"
    fraud_df.to_csv(target_path, index= False)
    print(f"Veri Seti Kaydedildi. Dosya Yolu: {target_path}")
    return fraud_df


if __name__ == "__main__":
    load_dataset(dataset_path= "D:/Datasets/creditcard.csv")