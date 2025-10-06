#%% import required libraries (Gerekli kutuphaneleri import et)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import numpy as np

from warnings import filterwarnings
filterwarnings('ignore')

#%% define eda function (EDA fonksiyonunu tanimla)

df_path = r"D:\Datasets\loan_dataset.csv"
def perform_eda():
    try:
        loan_df = pd.read_csv(df_path)
        print("Veri Seti Basariyla Yuklendi.")
        print("Ilk 5 Satir:\n", loan_df.head())
    except FileNotFoundError:
        raise FileNotFoundError("Dataset bulunamadi.Lutfen dosya yolunu kontrol edin.")
    
    except Exception as err:
        raise Exception(f"Veri yuklenirken bir hata olustu: {err}")
    
    # Show basic information about the dataset (Veri seti hakkinda temel bilgileri goster)
    print("Veri Setinin Boyutu:", loan_df.shape)
    print('Veri Setindeki Kayit Sayisi:', loan_df.shape[0])
    print("Veri Setindeki Sutun Sayisi:", loan_df.shape[1])
    print("Veri Setindeki Veri Tipleri:\n", loan_df.dtypes)
    print('Veri Setindeki Eksik Deger Toplami:\n',loan_df.isnull().sum().sum())
    
    
    target_col = 'loan_status'
    if target_col in loan_df.columns:
        print(f"Hedef Degisken '{target_col}' Kategorileri ve Frekanslari:\n", loan_df[target_col].value_counts())
    else:
        print(f"Hedef degisken '{target_col}' veri setinde bulunamadi.")
        
    # tarih sutunlarini datetime formatina cevir
    date_cols = ['effective_date', 'due_date', 'paid_off_time']
    for col in date_cols:
        if col in loan_df.columns:
            loan_df[col] = pd.to_datetime(loan_df[col], errors='coerce')
        else:
            print(f"Tarih sutunu '{col}' loan veri setinde bulunamadi")
            
    
    # Apply Feature extraction (Yeni ozellik turet)
    
    if {'due_date','effective_date'}.issubset(loan_df.columns):
        loan_df['planned_term_days'] = (loan_df['due_date'] - loan_df['effective_date']).dt.days
    else:
        print("planned_term_days ozelligi hesaplanamadi cunku 'due_date' veya 'effective_date' veri setinde bulunamadi.")    
    
    
    # Categorical feature analysis (Kategorik ozellik analizi)
    cat_cols = loan_df.select_dtypes(include = 'object').columns.tolist()
    num_cols = loan_df.select_dtypes(include=np.number).columns.tolist()
    '''num_cols = loan_df.select_dtypes(include = ['int32','float32']).columns.tolist()'''    
    print("Sayisal Degisken Analizi Yapiliyor...")
    for col in num_cols:
        print(f"Sayisal Degiskenler:\n {col} - Istatistikleri:\n {loan_df[col].describe().T}")
    
    
    print('Kategorik Degisken Analizi Yapiliyor...')
    for col in cat_cols:
        print(f"Kategorik Degiskenler:\n {col} - Frekanslari:\n {loan_df[col].value_counts(dropna = False)}")
        
    print('Veri Sizintisina (Data Leakage) Neden Olabilecek Sutunlar: Tarih Bilgisi Iceren Sutunlar Gosteriliyor...')    
    print("'paid_of_time','past_due_days' sutunlari izole et, egitimde kullanma!!!")
    print('Kardinal Sutunlar Model Egitiminden Cikariliyor...')
    print("'loan_id' sutunu kardinal sutun olarak kabul ediliyor... Model egitiminde kullanma!!!")
    


if __name__ == '__main__':
    perform_eda()


