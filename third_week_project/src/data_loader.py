#%% import required libraries (Gerekli kutuphaneleri import et)
import os 
import pandas as pd


#%% define load_data() function (load_data() fonksiyonunu tanimla)
def download_loan_data():
    """Loan datasetini indirir ve data klasorune kaydeder"""
    os.makedirs("data/raw",exist_ok = True)
    
    try:
        loan_df = pd.read_csv(r'D:\Datasets\loan_dataset.csv')
        print("Loan dataset indirildi.Ilk bes sutunu gosteriliyor...")
        print(loan_df.head())
    except FileNotFoundError:
        raise FileNotFoundError("Dataset bulunamadi.Lutfen dosya yolunu kontrol edin.")
    except Exception as e:
        raise Exception(f"Veri yuklenirken bir hata olustu: {e}")
    
    loan_df.to_csv('data/raw/loan_data.csv',index = False)
    
    return loan_df


#%% main function (ana fonksiyon)
if __name__ == "__main__":
    download_loan_data()
    
        