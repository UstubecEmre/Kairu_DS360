#%% import required libraries
import os 
import sys
import json 
import yaml 
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub 

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold #for cross validation
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor

import argparse
from datetime import datetime


import mlflow
import mlflow.sklearn



from model_preprocessing import Feature_Preprocessor, ImbalanceHandler
from model_evaluation import ModelEvaluater
from outlier_detection import main 

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings('ignore')


try:
    from model_explainability_fraud import SimpleModelExplainer
    logger.info("SimpleModelExplainer Import Edildi")
except ImportError as err:
    logger.error(f"Beklenmeyen Iceri Aktarma Hatasi: {err}")



class FraudDetectionPipeline():
    
    # constructor method
    def __init__(self, configuration_path = "config/configuration.yaml"):
        self.config = self._load_configuration(configuration_path)
        self._setup_logging()
        self._setup_mlflow()
        
        # Pipeline Elementleri (On isleme, modelleme, degerlendirme, kaydetme vb.)
        self.preprocessor = None
        self.explanier = None 
        self.models = {}
        self.evaluators = {}
        
        
        #X_train, X_test, y_train, y_test
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None 
        
        logger.info("Dolandiricilik Tespiti Pipeline'i Baslatiliyor")
        
        
    def _load_configuration(self, configuration_path):
        """Yapilandirma dosyasini yukler"""
        try:
            with open(configuration_path, mode  = "r") as conf_file:
                configuration = yaml.safe_load(conf_file) 
            return configuration
        except FileNotFoundError:
            logger.warning("Yapilandirma Dosyaniz Yuklenemedi. Varsayilanla Devam Edilecek")
            return self._get_default_config()
    
    def _get_default_config(self):
        """Varsayilan yapilandirmayi olusturur"""
        default_configuration = {
            "data": {
                'test_size':0.2
                ,'random_state': 42
            },
            'preprocessing':{
                'scaling_method': 'robust'
                ,'encoding_method': 'onehot'
            },
            'models':{
                'random_forest': {'random_state': 42,
                                  'n_estimators' : 200
                                  },
                'logistic_regression': {
                    'random_state': 42
                    
                },
                'isolation_forest': {
                    'random_state': 42,
                    'contamination': 0.05
                     
                },
            'evaluation': {
                  'min_roc_auc' : 0.75,
                  'min_prec_auc': 0.35  
                }
            }
        }   
        return default_configuration     
    
    def _setup_mlflow(self):
        """MLFlow yapilandirma dosyasi"""
        mlflow_configuration = self.config.get('mlflow', {})
        
        # takip icin
        tracking_uri = mlflow.config.get('tracking_uri', 'sqlite//mlflow.db')
        mlflow.set_tracking_uri(tracking_uri)
        
        # gozlemlerini olusturalim
        experiment_name = mlflow.config.get('experiment_name', 'fraud_detection')
        mlflow.set_experiment(experiment_name=experiment_name)

        # otomatik loglamayi saglayalim
        if mlflow.config.get('autolog', {}).get('sklearn', True):
            mlflow.sklearn.autolog()
        
        logger.info(f"MLFlow Yapilandirilmasi Basarili Bir Sekilde Gerceklesti: {experiment_name}")
        
        
    def load_fraud_data_from_kagglehub(self, synthetic = True, data_path = None, dowload_with_kagglehub = False):
        
        """download_with_kagglehub argumani True ise kagglehub uzerinden fraud detection veri setini yukler"""
        if dowload_with_kagglehub:
            logger.info("KaggleHub Uzerinden Dolandiricilik Tespiti (Fraud Detection) Veri Seti Indiriliyor")
            
            try:
                import kagglehub
                path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
                fraud_csv = os.path.join(path, 'creditcard.csv')
                
                # dosya var mi? Varsa uzerine yaz, yoksa olustur
                if os.path.exists(fraud_csv):
                    fraud_data = pd.read_csv(fraud_csv)
                    logger.info(f"KaggleHub Uzerinden Veri Seti Indirildi. Veri Seti Boyutu: {fraud_data.shape}")  
                else:
                    logger.warning(f"Veri Seti KaggleHub Uzerinden Indirilemedi. Bilgisayariniza Indirebilirsiniz:")
            except Exception as err:
                logger.error(f"KaggleHub Uzerinden Indirme Basarisiz: {err}")  
                #Dileyen yapay veri ureterek bu islemleri gerceklestirebilir, Program askida kalmaz.
        else:
            try:
                fraud_data = pd.read_csv(data_path)
                logger.info(f"Local'deki Yol Uzerinden Veri Seti Indiriliyor")
            except FileNotFoundError:
                logger.error("Dosya Yolu Bulunamadi. Lutfen Dosya Yolunu Kontrol Ediniz!!!")
            except Exception as err:
                logger.error(f"Beklenmeyen Bir Hata Olustu: {err}")
        
        # veri setini incele
        self._validate_data(fraud_data)
        
        
        # Veri setini X ve y olarak ayiralim (split)
        X = fraud_data.drop('Class', axis= 1) # hedef degiskeni cikar
        y = fraud_data['Class']
        
        
        test_size = self.config['fraud_data']['test_size']
        random_state = self.config['fraud_data']['random_state']
        
        # train test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size= test_size,
            random_state=random_state,
            stratify = y if self.config['fraud_data'].get('stratify', True) else None 
        )    

        logger.info(f"Veri Seti Yuklendi. Egitim Seti Boyutu: {len(self.X_train)}\nTest Seti Boyutu: {len(self.X_test)}")
        logger.info(f"Sinif Dagilimlari:\n Egitim Seti Orani: {np.bincount(self.y_train)}\nTest Seti: {np.bincount(self.y_test)}")
        
    
    def _validate_data(self, data):
        """Veri setinin kalitesini ve dogrulunu test eder"""
        validation_config = self.config.get('data', {}).get('validation',{})
        
        # Gerekli sutunlari alalim. Islem tutari, dolandiricilik sinifi, zaman
        required_cols = validation_config.get('required_columns', ['Amount', 'Time','Class'])
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            logger.error(f"Eksik Veri Bulunmaktadir: {missing_cols}")
        
        
        # hedef degiskeni sayisal hale getir
        if 'Class' in data.columns:
            valid_classes = validation_config.get('class_values', [0, 1])
            invalid_classes = data['Class'].unique() # tekil degerleri
            invalid_classes = [col for col in invalid_classes if col not in valid_classes]
            
            if invalid_classes:
                logger.error(f"Gecersiz Sinif Bulunmaktadir: {invalid_classes}")
        
        
        # Amount degiskenini incele => 0 ile 1 Milyon araliginda tutalim
        if 'Amount' in data.columns:
            min_amount = validation_config.get('amount_min', 0)
            max_amount = validation_config.get('amount_max', 1000000)
            
            if data['Amount'].min() < min_amount or data['Amount'].max() > max_amount:
                logger.error(f"Beklenen Degerler Icerisinde Yer Almamaktadir. Beklenen Aralik: [{min_amount}, {max_amount}]")
                