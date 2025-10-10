# import required libraries (Onemli kutuphaneleri import et)
import os
import numpy as np
import pandas as pd
from pathlib import Path
# import kagglehub 

DATA_DIR = "data/raw"

def ensure_raw_data():
    try: 
        os.makedirs(DATA_DIR, exist_ok= True)
        print("Klasor Olusturuldu")
    except Exception as err:
        print(f"Beklenmeyen Bir Hata Olustu: {err}")


def load_dataset(dataset_path:str):
    ensure_raw_data()
    if not isinstance(dataset_path,str) or not dataset_path:
        print("Veri Setinizin Yolu String Formatta Olmalı ve Boş Olmamalıdır")
        return False
    try:
       # dataset_path = r"D:/Datasets/creditcard.csv"
        fraud_df = pd.read_csv(dataset_path)
        print("Fraud DataFrame'i Basarili Bir Sekilde Yuklendi.")
    except FileNotFoundError:
        raise FileNotFoundError(f"Dosya Yolu Bulunamadi! Lutfen Dogru Bir Dosya Yolu Giriniz")
    except Exception as err:
        raise Exception(f"Beklenmeyen Bir Hata Gerceklesti: {err}")
    
    target_path = Path(DATA_DIR) / "creditcard_fraud.csv"
    fraud_df.to_csv(target_path, index= False)
    return fraud_df


if __name__ == "__main__":
    load_dataset(dataset_path= "D:/Datasets/creditcard.csv")