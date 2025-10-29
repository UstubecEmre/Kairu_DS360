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