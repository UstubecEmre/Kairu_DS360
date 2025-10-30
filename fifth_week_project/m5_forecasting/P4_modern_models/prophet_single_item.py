""" 
Prophet, Meta (eski bilinen ismiyle Facebook) tarafindan gelistirilmis, modern, kullanici dostu bir yontemdir.
ARIMA modele gore mevsimsellikleri, tatil gunlerini de hesaba katarak modelleme gerceklestirmemize olanak saglamaktadir 
"""


#%% Gerekli Kutuphanelerin Dahil Edilmesi (import required libraries)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import json
from datetime import timedelta, datetime

from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

#%% Prophet dahil edelim

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    print(f"Prophet Kutuphanesi Bulunamadi. Lutfen Dahil Ediniz")
    PROPHET_AVAILABLE = False


#%% 
class ProphetModelSingleItemForecaster():
    def __init__(self, item_id = None, artifacts_path:str = r'D:\Kairu_DS360_Projects\fifth_week_project\m5_forecasting\artifacts'):
        self.artifacts_path = os.path.abspath(artifacts_path)
        self.item_id = item_id
        self.train_series = None
        self.valid_series = None
        self.prophet_model = None
        self.forecast_df = None
        self.metrics = {}
        self.arima_metrics = {}
        
        # Kaydedilmeleri icin klasor olusturalim
        
        os.makedirs(f'{artifacts_path}/models', exist_ok = True)
        os.makedirs(f'{artifacts_path}/predictions', exist_ok = True)
        os.makedirs(f'{artifacts_path}/figures', exist_ok = True)
    
    
    def load_arima_results(self):
        """Prophet ve ARIMA Modelleri Karsilastirilmasi Icin Gerekli Sonuclari Yukler"""
        print("Karsilastirma Yapabilmek Amaciyla ARIMA Yontemi Sonuclari Yukleniyor...")
        try:
            arima_files = [file for file in os.listdir(f'{self.artifacts_path}/models') if file.startswith('arima_')]
            
            # dosya var mi?
            if not arima_files:
                print("Maalesef ARIMA Modeli Bulunamadi. Lutfen 'arima_single_item.py' Python Dosyasini Calistirin")
                return None 
            
            arima_file = arima_files[0]
            self.item_id = arima_file.replace('arima_', '').replace('.pkl', '')
            print(f"ARIMA Modelinden 'item_id' Bilgisi Aliniyor. 'item_id': {self.item_id}")
            
            
            print("ARIMA Raporunuz Getiriliyor...")
            arima_report_path = f'{self.artifacts_path}/predictions/arima_report_{self.item_id}.json'
            # okuma modunda acalim
            with open(arima_report_path, 'r') as file:
                arima_report = json.load(file)
            
            
            # Metriklere dict uzerinden ulasalim
            self.arima_metrics = arima_report['metrics']
            print("ARIMA Modeli Degerlendirme Metrikleri Getiriliyor...")
            print(f"Ortalama Mutlak Hata Degeri (MAE): {self.arima_metrics['MAE']:.4f}")
            print(f"Simetrik Ortalama Mutlak Yuzdesel Hata Degeri (sMAPE): {self.arima_metrics['sMAPE']:.4f}")
        
            return arima_report
        
        except Exception as err:
            raise Exception(f"Karsilastirilacak ARIMA Modeli Getirilirken Beklenmeyen Bir Hata Olustu: {err}")


    def load_time_series(self):
        pass