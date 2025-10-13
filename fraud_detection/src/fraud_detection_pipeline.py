#%% import required libraries

import sys
import json 
import yaml 
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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