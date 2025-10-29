""" 
ARIMA (Aristmetic Integrated Moving Average) model tek bir urun icin satis tahmini yapar.
Bu model zaman serisi verilerinde trend ve mevsimsellik gibi ozellikleri yakalamak icin kullanilir.
"""

#%% Gerekli Kutuphaneleri Dahil Edelim (import required libraries)
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from itertools import product
from datetime import datetime, timedelta

import warnings
warnings.filterwarnings("ignore")

# statsmodels kutuphanesinden ARIMA modelini ve performans metriklerini import edelim
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# model degerledirme metrikleri (regresyon oldugu icin MSE, MAE)
from sklearn.metrics import mean_squared_error, mean_absolute_error


class ARIMAModelSingleItemForecaster:
    """ARIMA modeli kullanarak tek bir urun icin satis tahmini yapar."""
    
    
    def __init__(self, artifacts_path : str = './artifacts'):
        self.artifacts_path = os.path.abspath(artifacts_path)
        self.item_id = None
        self.ts_data = None
        self.train_series = None
        self.valid_series = None
        self.model = None
        self.forecast = None
        self.best_params = None
        self.metrics = {}
        self.create_artifacts_dirs()
        
        
    def create_artifacts_dirs(self):
        """Model artifact dizinlerini olusturur."""
        model_dir = os.path.join(self.artifacts_path, 'models')
        preds_dir = os.path.join(self.artifacts_path, 'predictions')
        figures_dir = os.path.join(self.artifacts_path, 'figures')
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(preds_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)
        """ 
        for sub_dir in [model_dir, preds_dir, figures_dir]:
            os.makedirs(os.path.join(self.artifacts_path, sub_dir), exist_ok=True)
        """
    def select_top_item(self):
        """En cok satan urunu secer"""
        print("[INFO] En Cok Satan Urun Seciliyor...")
        try:
            # egitim verisini yukleyelim
            train_df = pd.read_csv(r"D:\Kairu_DS360_Projects\fifth_week_project\m5_forecasting\artifacts\datasets\train.csv")
            
            # urun bazinda toplam satislari hesaplayalim
            item_totals = train_df.groupby('item_id')['sales'].sum().sort_values(ascending=False)
            print("Urun Bazinda Toplam Satislar Hesaplandi.")
            for idx, (item, total) in enumerate(item_totals.items()):
                print(f"{idx+1}. Urun: {item}, Toplam Satis: {total}")
                
            # en cok satan urunu alalim
            self.item_id = item_totals.index[0]
            top_sales = item_totals.iloc[0]
            
            print(f"En Cok Satilan Urun: {self.item_id} \nToplam Satis Adedi: {top_sales}")
        except FileNotFoundError:
            raise FileNotFoundError("Egitim verisi bulunamadi. Lutfen create_m5_subset.py Kullanarak Veri Setini Hazirlayin.")

        return self.item_id
    
    def load_time_series(self):
        """Secilen urun icin zaman serisi verisini yukler."""
        print(f"[INFO] Zaman Serisi Verisi Yukleniyor: {self.item_id}")
        try:
            # egitim verisini yukleyelim
            train_df = pd.read_csv(r"D:\Kairu_DS360_Projects\fifth_week_project\m5_forecasting\artifacts\datasets\train.csv")
            valid_df = pd.read_csv(r"D:\Kairu_DS360_Projects\fifth_week_project\m5_forecasting\artifacts\datasets\validation.csv")
        # secilen urunun zaman serisi verisini alalim
        
            item_train = train_df[train_df['item_id'] == self.item_id]['sales'].copy()
            item_valid = valid_df[valid_df['item_id'] == self.item_id]['sales'].copy()
            
            print(f"Egitim Verisi Zaman Araligi: {item_train.index.min()} - {item_train.index.max()}")
            print(f"Dogrulama Verisi Zaman Araligi: {item_valid.index.min()} - {item_valid.index.max()}")
            print(f"Egitim Verisi Gun Sayisi: {len(item_train)}")
            print(f"Dogrulama Verisi Gun Sayisi: {len(item_valid)}")
            
            
            # Serilerimizi koruyalim
            self.train_series = item_train
            self.valid_series = item_valid
            
            # Verilerimizi birlestirelim
            self.ts_data = pd.concat([item_train, item_valid])
            
            print(f"[INFO] Toplam Gunluk Veri Sayisi: {len(self.ts_data)}")
            
            print("Temel Istatistik Bilgileri Getiriliyor...")
            print(f"Ortalama: {self.train_series.mean():.4f}")
            print(f"Standart Sapma: {self.train_series.std():.4f}")
            print(f"En Dusuk Satis: {self.train_series.min()}")
            print(f"En Yuksek Satis: {self.train_series.max()}")
            print(f"Hic Satis Yapilmayan Gun Sayisi: {(self.train_series == 0).sum()}")

        except FileNotFoundError:
            raise FileNotFoundError("Belirtilen Dosya Yollarinda Ilgili Klasorler Bulunamadi!!!")
        
        except Exception as err:
            raise Exception(f"Beklenmeyen Bir Hata Olustu: {err}")
               
        return self.ts_data
    
    
    def test_stationarity(self, series, title = 'Series'):
        """Stationarity (ADF) Test Uygular"""
      
        print(f"{title} Icin Stationarity Test Gerceklesitiriliyor...")
        
        adf_result = adfuller(series.dropna())
        print(f"ADF Istatistikleri: {adf_result[0]:.6f}")
        print(f"p-Degeri: {adf_result[1]:.6f}")
        print(f"Kritik Degerler: ")
        for key, value in adf_result[4].items():
            print(f"{key}: {value:.4f}")
            
        # Sonuc yorumu
        if adf_result[1] <= 0.05:
            print(f"Duragan (Stationary) p-Degeri < 0.05")
            is_stationary = True
        else:
            print(f"Duragan Degil (Non-Stationary) p-Degeri > 0.05")
            is_stationary = False
            
        return {"is_stationary": is_stationary, "p_value": adf_result[1], "adf_stat": adf_result[0]}

    
    
    def determine_d_parameter(self):
        """d(Differencing order) parametresini belirle"""
        print("ARIMA Yontemi Icin d Parametresi Belirleniyor")
        
        series = self.train_series.copy()
        d = 0
        max_d =  2 # maksimum 2 kez 
        
        # orijinal serinin duraganligini test edelim
        is_stationary, p_value = self.test_stationarity(series, f'Orijinal: (d = {d})')
        
        # duragansa
        if is_stationary:
            print(f"{d} Parametre Ayari Gerekmiyor. Duragan...")
            return d
        
        
        for d in range(1, max_d + 1):
            diff_series = series.diff(d).dropna()
            
            if len(diff_series) < 50:
                print(f"d = {d} Degeri Icin Yeterli Veri Kalmamaktadir. d = {d - 1} Kullanilacaktir")
                return d - 1
            
            is_stationary, p_value = self.test_stationarity(diff_series,f"Ayarlanan d degeri: {d}")
            if is_stationary:
                print(f"d Degeri Olarak {d} Belirlendi")
                return d
        
        print(f"Optimal d Degeri Bulunamadi, d = 1 Olarak Belirlenecektir")
        return 1
    
    def grid_search_arima(self, max_p = 2, max_q = 2):
        """ARIMA Yontemi Icin Parametrelerin Grid Search Kullanilarak Optimize Edilmesi"""
        print("ARIMA Grid Search Islemi Gerceklestiriliyor...")
        
        d = self.determine_d_parameter()
        
        # grid search icin parametreler
        p_values = range(0, max_p + 1)
        q_values = range(0, max_q + 1)
        
        print(f"p Degerleri: {list(p_values)}")
        print(f"q Degerleri: {list(q_values)}")
        print(f"Kullanilan d Degeri: {d}")
        
        
        best_aic = float('inf')
        best_params = None
        results = [] 

        
        total_combinations = len(q_values) * len(p_values)
        current_combination = 0
        
        for p in p_values:
            for q in q_values:
                current_combination += 1
                try:
                    print(f"ARIMA Yontemi Icin ({p}, {q}, {d}) Parametreleri Deneniyor... ({current_combination}/{total_combinations}")
                    # modeli egitelim
                    model = ARIMA(self.train_series, order = (p, q, d))
                    fitted_model = model.fit()
                    
                    aic = fitted_model.aic
                    bic = fitted_model.bic
                    
                    results.append(
                        {
                            'p':p
                            ,'d': d
                            ,'q': q
                            ,'AIC': aic
                            ,'BIC': bic
                            ,'converged': fitted_model.mle_retvals['converged'] if hasattr(fitted_model, 'mle_retvals') else True
                            
                        }
                    )
                    
                    if aic < best_aic:
                        best_aic = aic
                        best_params = (p, d, q)
                        print(f"En Iyi AIC = {aic:.4f}")
                except Exception as err:
                    print(f"Beklenmeyen Bir Hata Olustu: {str(err)[:25]}")
                    results.append(
                        {
                            'p': p
                            ,'q': q
                            ,'d' : d
                            ,'AIC': np.inf
                            ,'BIC': np.inf
                            ,'converged': False
                            
                        }
                    )
        # DataFrame'e donusturelim
        results_df = pd.DataFrame(results)
        print("Grid Search Sonuclari Gosteriliyor...")
        print(f"En Iyi Parametreler: ARIMA{best_params}")
        print(f"En Iyi AIC Degeri: {best_aic:.4f}")
        
        # En iyi 3 sonuc
        top_results = results_df[results_df['converged']].nsmallest(3, 'AIC')
        print("En Iyi 3 Model")
        
        
        for idx, (_, row) in enumerate(top_results.iterrows(), 1):
            print(f"{idx}. ARIMA({int(row['p'])},{int(row['d'])},{int(row['q'])}) - AIC: {row['AIC']:.2f}")
        
        self.best_params = best_params
        return best_params, results_df