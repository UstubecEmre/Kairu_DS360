#%% import required libraries (Gerekli kutuphaneleri import et)
import os 
import pandas as pd


#%% define load_data() function (load_data() fonksiyonunu tanimla)
def load_loan_data(save_to_disk = True):
    """Loan datasetini indirir ve data klasorune kaydeder"""
    
    
    try:
        loan_df = pd.read_csv(r'D:\Datasets\loan_dataset.csv')
        print("Loan dataset indirildi.Ilk bes sutunu gosteriliyor...")
        print(loan_df.head())
    except FileNotFoundError:
        raise FileNotFoundError("Dataset bulunamadi.Lutfen dosya yolunu kontrol edin.")
    except Exception as e:
        raise Exception(f"Veri yuklenirken bir hata olustu: {e}")
    
    if save_to_disk:
        base_dir = r"D:\Kairu_DS360_Projects\third_week_project"
        os.makedirs(os.path.join(base_dir, "data/raw"), exist_ok=True)
        loan_df.to_csv(os.path.join(base_dir, "data/raw/loan_data.csv"), index=False)
   
    return loan_df

loan_df = load_loan_data(save_to_disk=True)
"""#%% main function (ana fonksiyon)
if __name__ == "__main__":
    load_loan_data(save_to_disk=True)
    
"""     