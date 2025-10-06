#%% import required libraries (Gerekli kutuphaneleri import et)
import os 
import pandas as pd


#%% define load_data() function (load_data() fonksiyonunu tanimla)
"""def load_loan_data(save_to_disk = True):
    #Loan datasetini indirir ve data klasorune kaydeder
    
     dvc.yaml ile uyumlu hale getirilmeli. Burada hata verebilir!!!
    try:
        loan_df = pd.read_csv(r'D:/Datasets/loan_dataset.csv')
        print("Loan dataset indirildi.Ilk bes sutunu gosteriliyor...")
        print(loan_df.head())
    except FileNotFoundError:
        raise FileNotFoundError("Dataset bulunamadi.Lutfen dosya yolunu kontrol edin.")
    except Exception as e:
        raise Exception(f"Veri yuklenirken bir hata olustu: {e}")
    
    if save_to_disk:
        base_dir = r"D:/Kairu_DS360_Projects/third_week_project/loan_data_analysis"
        os.makedirs(os.path.join(base_dir, "data/raw"), exist_ok=True)
        loan_df.to_csv(os.path.join(base_dir, "data/raw/loan_data.csv"), index=False)
   
    return loan_df
    """ 
    
def load_loan_data(save_to_disk=True):
    """Loan datasetini indirir ve data/raw klasorune kaydeder"""

    # DVC’nin takip edeceği yer
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    raw_dir = os.path.join(data_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    # Raw data file path
    raw_file_path = os.path.join(raw_dir, "loan_data.csv")

    # Eğer dataset hali hazırda yoksa dış kaynaktan yükle
    if not os.path.exists(raw_file_path):
        external_path = r"D:/Datasets/loan_dataset.csv"  # Orijinal dataset yolun
        if not os.path.exists(external_path):
            raise FileNotFoundError(f"Dataset bulunamadi: {external_path}")
        df = pd.read_csv(external_path)
        if save_to_disk:
            df.to_csv(raw_file_path, index=False)
    else:
        df = pd.read_csv(raw_file_path)

    print("Loan dataset başarıyla yüklendi. İlk 5 satır:")
    print(df.head())
    return df

#%% main function (ana fonksiyon)
if __name__ == "__main__":
    load_loan_data(save_to_disk=True)
    
     