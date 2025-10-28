#%% import required libraries
import os 
import sys
import json 
import yaml
import joblib 
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
from outlier_detection import main,OutlierDetection, OutlierDetector

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
        self.explainer = None 
        self.models = {}
        self.evaluators = {}
        
        
        #X_train, X_test, y_train, y_test
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None 
        
        logger.info("Dolandiricilik Tespiti Pipeline'i Baslatiliyor")
    
    def _setup_logging(self):
        """Logging ayarlamasi"""
        global logger # global logger tanimla
        
        # yapilanmayi ayarla
        log_config = self.config.get('logging', {})
        
        # ozelliklerini al
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        
        # formatini belirle
        log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        
        logging.basicConfig(level=log_level, format=log_format)
        
        logger = logging.getLogger(__name__)
        
        # File logging
        if log_config.get('file_logging', False):
            # eger file_logging yoksa logs isminde bir klasor olustur
            os.makedirs('logs', exist_ok=True)
            
            file_handler = logging.FileHandler(log_config.get('log_file', 'logs/pipeline.log'))
            # formatini ayarla
            file_handler.setFormatter(logging.Formatter(log_format))
            
            logger.addHandler(file_handler)
        
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
                ,'raw_data_path': 'data/raw/creditcard_fraud.csv'
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
        tracking_uri = mlflow_configuration.get('tracking_uri', 'sqlite///mlflow.db')
        mlflow.set_tracking_uri(tracking_uri)
        
        # gozlemlerini olusturalim
        experiment_name = mlflow_configuration.get('experiment_name', 'fraud_detection')
        mlflow.set_experiment(experiment_name)

        # otomatik loglamayi saglayalim
        if mlflow_configuration.get('autolog', {}).get('sklearn', True):
            mlflow.sklearn.autolog()
        
        logger.info(f"MLFlow Yapilandirilmasi Basarili Bir Sekilde Gerceklesti: {experiment_name}")
        
    
    
    
    
    def load_fraud_data_from_kagglehub(self, synthetic = False, data_path = None, download_with_kagglehub = False):
        
        """download_with_kagglehub argumani True ise kagglehub uzerinden fraud detection veri setini yukler"""
        fraud_data = None 
        # configuration dosyasindan parametreleri al
        DEFAULT_DATA_PATH = Path(self.config['data']['raw_data_path'])
        
        
        if download_with_kagglehub:
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
        if fraud_data is None:
            path_to_use = data_path if data_path is not None else DEFAULT_DATA_PATH
            
            if path_to_use is None:
                logger.error("Lutfen Gecerli Bir Veri Dosyasi Yolu Belirleyiniz")
            else:
                try:
                    fraud_data = pd.read_csv(path_to_use)
                    logger.info(f"Veri Seti Yuklendi. Veri Seti Boyutu: {fraud_data.shape}")
                except FileNotFoundError:
                    logger.error(f"Veri Dosyasi Bulunamadi. Lutfen Gecerli Bir Yol Belirleyiniz: {path_to_use}")
                except Exception as err:
                    logger.error(f"Veri Yukleme Sirasinda Beklenmeyen Hata Olustu: {err}")
        
        if fraud_data is None:
            if synthetic:
                raise ValueError("Yapay Veri Uretimi Mevcut Degildir.")
            else:
                raise ValueError("Veri Seti Yuklenemedi. Lutfen Gecerli Bir Veri Dosyasi Yolu Belirleyiniz")
        # veri setini incele
        self._validate_data(fraud_data)
        
        
        # Veri setini X ve y olarak ayiralim (split)
        X = fraud_data.drop('Class', axis= 1) # hedef degiskeni cikar
        y = fraud_data['Class']
        
        
        test_size = self.config['data']['test_size']
        random_state = self.config['data']['random_state']
        
        # train test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size= test_size,
            random_state=random_state,
            stratify = y if self.config['data'].get('stratify', True) else None 
        )    

        logger.info(f"Veri Seti Yuklendi. Egitim Seti Boyutu: {len(self.X_train)}\nTest Seti Boyutu: {len(self.X_test)}")
        logger.info(f"Sinif Dagilimlari:\n Egitim Seti Orani: {np.bincount(self.y_train)}\nTest Seti: {np.bincount(self.y_test)}")
        return fraud_data
    
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
                
        logger.info("Veri Dogrulama Gerceklestirildi")
        
    
    def preprocess_data(self):
        """Veri on isleme adimlarini gerceklestirir. Scaling, encoding..."""
        preprocessing_config = self.config.get('preprocessing', {})
        
        self.preprocessor = Feature_Preprocessor(
            #"scaling_method": 'robust'
            scaling_method= preprocessing_config.get('scaling_method','robust'),
            encoding_method = preprocessing_config.get('encoding_method', 'onehot')
        )
        
        # 
        train_fraud_data = pd.concat([self.X_train, self.y_train])
        train_fraud_processed = self.preprocessor.fit_transform_model(
            train_fraud_data,
            target_col = 'Class' 
        )
        
        test_fraud_data = pd.concat([self.X_test, self.y_test])
        test_fraud_processed = self.preprocessor.transform_model(
            test_fraud_data,
            target_col = 'Class'
        )   
        
        
        # hedef degiskeni cikar
        self.X_train_processed = train_fraud_processed.drop('Class', axis = 1)
        self.y_train_processed = train_fraud_processed['Class']
        
        
        self.X_test_processed = test_fraud_processed.drop('Class', axis = 1)
        self.y_test_processed = test_fraud_processed['Class']
        
        logger.info("Veri On Isleme Basarili Bir Sekilde Gerceklestirildi")
        
        # Dengesiz veri seti soz konusuysa SMOTE, ADASYN SMOTETek vb. yontemler uygula
        imbalance_config = self.config.get('imbalance', {})
        method = imbalance_config.get('method','smote')
        
        if method == 'smote':
            self.X_train_balanced, self.y_train_balanced = ImbalanceHandler.apply_smote(
                X = self.X_train_processed,
                y = self.y_train_processed,
                #sampling_strategy= 'auto',
                # random_state = 42
                sampling_strategy = imbalance_config.get('sampling_strategy', 'auto'),
                random_state = imbalance_config.get('random_state', 42)
            )
        elif method == 'adasyn':
            self.X_train_balanced, self.y_train_balanced = ImbalanceHandler.apply_adasyn(
                X = self.X_train_processed,
                y = self.y_train_processed,
                #sampling_strategy= 'auto',
                # random_state = 42
                sampling_strategy = imbalance_config.get('sampling_strategy', 'auto'),
                random_state = imbalance_config.get('random_state', 42)
            )
        elif method == 'smotetomek':
            self.X_train_balanced, self.y_train_balanced = ImbalanceHandler.apply_smotetomek(
                X = self.X_train_processed,
                y = self.y_train_processed,
                #sampling_strategy= 'auto',
                # random_state = 42
                sampling_strategy = imbalance_config.get('sampling_strategy', 'auto'),
                random_state = imbalance_config.get('random_state', 42)
            )
        else:
            self.X_train_balanced = self.X_train_processed
            self.y_train_balanced = self.y_train_processed
        logger.info(f"Veri Duzensizligi Azaltildi: {len(self.X_train_balanced)}")
        
    def train_models(self):
        """Modelleri egitir"""
        logger.info("Modeller Egitiliyor")
        models_config = self.config.get('models', {})
        
        with mlflow.start_run():
            mlflow.log_params(
                {
                    'data_size': len(self.X_train_balanced),
                    'n_features': self.X_train_balanced.shape[1], # sutunlari 
                    'preprocessing_method': self.config.get('preprocessing', {}).get('scaling_method', 'robust')
                    
                }
             )
            
        
        # modelleri isim ve parametreleri ile donguye sok
        for model_name, model_params in models_config.items():
            logger.info(f"{model_name} modeli egitiliyor")
            with mlflow.start_run(nested= True):
            
                if model_name == 'random_forest':
                    model = RandomForestClassifier(**model_params)
                
                elif model_name == 'logistic_regression':
                    model = LogisticRegression(**model_params)
                
                elif model_name == 'lof':
                    model = LocalOutlierFactor(**model_params)
                    
                elif model_name == 'isolation_forest':
                    model = IsolationForest(**model_params)
                else:
                    logger.warning(f"Gecersiz Bir Model Ismi Girdiniz")
                    continue # program hata vermeden devam etsin.
                
                # Aykiri deger tespiti icin LocalOutlierFactor ve IsolationForest
                if model_name in ['lof', 'isolation_forest']:
                    model.fit(self.X_train_balanced)  #dengeli veri ile egit
                else:
                    model.fit(self.X_train_balanced, self.y_train_balanced)   
                    
                self.models[model_name] = model
                
                if model_name not in ['lof', 'isolation_forest']: # RandomForestClassifier, LogisticRegression
                    # Capraz dogrulama yap
                    cv_scores = cross_val_score(
                        model,
                        self.X_train_balanced,
                        self.y_train_balanced,
                        cv = 5, # 10 da yapilabilir
                        scoring = 'roc_auc'
                    )
                    mlflow.log_metric(f"{model_name}_cross_validation_roc_auc_mean", cv_scores.mean())
                    mlflow.log_metric(f"{model_name}_cross_validation_roc_auc_std", cv_scores.std())    
                    logger.info(f"{model_name} Modelinin Capraz Dogrulama ROC-AUC Egri Basari Ortalamasi: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

                else:
                    mlflow.log_param("unsupervised_model", True)
                    logger.info(f"{model_name} Modelinin Egitimi Tamamlandi (Denetimsiz Ogrenme ile)")
        
        logger.info("Model Egitimleri Gerceklestirildi")
        
        
    def evaluate_models(self):
        """Egitilen modelleri degerlendirir"""
        logger.info("Model Degerlendirilmesi Yapiliyor")
        evaluation_config = self.config.get('evaluation', {})
            
        for model_name, model in self.models.items():
            logger.info(f"{model_name} Degerlendiriliyor")

                
            evaluator = ModelEvaluater(
                model = model,
                model_name = model_name
            )
                
            
        # Aykiri deger icin kullanilan Makine Ogrenmesi Algoritmalari ise ['lof','iso']
            if model_name in  ['isolation_forest', 'lof']:
                detector = OutlierDetector()
                    
                if hasattr(self, "preprocessor") and hasattr(self.preprocessor, "scaler"):
                    detector.scaler = self.preprocessor.scaler
                else:
                    logger.warning("Scaler bulunamadı, OutlierDetector normalize edilmeden çalışacak.")
               
                if model_name == 'isolation_forest':
                    detector.isolation_forest = model
                    predictions = detector.predict_isolation_forest(self.X_test_processed)
                        
                else:
                    detector.lof = model
                    predictions = detector.predict_lof(self.X_test_processed)
                
                y_pred_proba = predictions.astype(float)
                results = evaluator.calc_metrics(
                    self.X_test_processed,
                    self.y_test_processed,
                    y_pred_proba
                )
            else:
                results = evaluator.calc_metrics(
                    self.X_test_processed,
                    self.y_test_processed
                )
                
            # model degerlendirmek icin modelin ismini ver
            self.evaluators[model_name] = evaluator
                
                
            # MLFlow'da metrikleri kaydet
            with mlflow.start_run(nested= True):
                mlflow.log_params(
                    model.get_params() if hasattr(model, 'get_params') else {}
                )
                """ liste elemanlari hata verir, sadece sayisal olanlari kaydet
                mlflow.log_metrics(
                    {
                        f"{model_name}_accuracy": results['accuracy'],
                        f"{model_name}_precision": results['precision'],
                        f"{model_name}_recall": results['recall'],
                        f"{model_name}_f1": results['f1'],
                            
                        f"{model_name}_confusion_matrix": results['confusion_matrix'],
                        f"{model_name}_roc_auc": results['roc_auc'],
                        f"{model_name}_precision_curve": results['f1'],
                        f"{model_name}_recall_curve": results['f1'],
                        f"{model_name}_threshold": results['threshold']
                      
                    }
                )"""
                mlflow.log_metrics(
                    {f"{model_name}_accuracy": results['accuracy'],
                    f"{model_name}_precision": results['precision'],
                    f"{model_name}_recall": results['recall'],
                    f"{model_name}_f1": results['f1'],
                    f"{model_name}_roc_auc": results['roc_auc']
                    }
                )
                # karmaşıklık için JSON olarak kaydet
                mlflow.log_dict(results, f"{model_name}_detailed_metrics.json")
                # RandomForestClassifier() veya LogisticRegression()
                if model_name not in ['isolation_forest', 'lof']:
                    mlflow.sklearn.log_model(model, f"{model_name}_model")
                    
                evaluator.print_evaluation()
                    
                # Performans esik degerlerini kontrol et min_auc_roc = 0.75, min_prec_auc = 0.3
                min_roc_auc = evaluation_config.get('min_roc_auc', 0.75)
                min_prec_auc = evaluation_config.get('min_prec_auc', 0.35)
                    
                if results['roc_auc'] < min_roc_auc:
                    logger.warning(f"{model_name} Modelinin ROC-AUC Degeri: {results['roc_auc']:.4f} Belirlenen Basari Esik Degerinden: {min_roc_auc} Dusuktur")     
                        
                if results['prec_auc'] < min_prec_auc:
                    logger.warning(f"{model_name} Modelinin Prec-AUC Degeri: {results['prec_auc']:.4f} Belirlenen Basari Esik  Degerinden: {min_prec_auc} Dusuktur")
                        
            logger.info("Model Degerlendirilmesi Tamamlandi")
                
                
                
    def explain_model(self, model_name = 'random_forest'):
        """Model Aciklanabilirligini saglar"""
        if SimpleModelExplainer is None:
            logger.warning("SimpleModelExplainer Bos Gecilemez")
            return None, None 
                        
        if model_name not in self.models:
            logger.error(f"{model_name} Modeli Gecerli Degildir")
            return False 
                    
        logger.info(f"{model_name} Modeli Aciklanabilir Hale Getiriliyor...")
                    
        self.explainer = SimpleModelExplainer(
                    self.models[model_name],
                    self.X_train_balanced,
                    feature_names= list(self.X_train_processed.columns),
                    class_names = ['Normal', 'Fraud']
                )
                    
        # SHAP analizi
        explainer_config = self.config.get('explainability', {}).get('shap', {})
        self.explainer.initialize_shap(
            explainer_type = explainer_config.get('explainer_type', 'auto'),
            # max_evals = explainer_config.get('max_evals', 100) 
        )
                    
        shap_values, X_sample = self.explainer.compute_shap_values(
                self.X_test_processed,
                max_samples=explainer_config.get('max_samples', 100)
        )
        self.explainer.plot_shap_summary(X_sample)
                    
        # Global degiskenlerin onem derecesi
        importance = self.explainer.global_feature_importances(X_sample)
                    
        # Anomali oruntulerini bulalim
        fraud_patterns = self.explainer.analyze_fraud_pattern(
        # self.X_test_processed, self.y_test_processed
            X_sample, 
            self.y_test_processed[:len(X_sample)]
        )
                    
        logger.info("Model Aciklanabilirligi Tamamlandi")
        return importance, fraud_patterns
                

    def save_models(self, save_path = "models/"):
        """Modelleri ve On Isleme Adimlarini Models Klasor  """
        try:
            os.makedirs(save_path, exist_ok= True)
            logger.info(f"Dosya Olusturuldu. Dosya Yolu: {save_path}")
        except Exception as err:
            logger.error(f"Dosya Olusturulamadi: {err}")
            
        # on isleme adimlarini kaydet
        joblib.dump(
            self.preprocessor,
            os.path.join(save_path, 'preprocessor.pkl')
        )
        logger.info("Veri On Isleme Adimlari Kaydedildi")
        
        # Modelleri dongu yardimi ile kaydet
        for model_name, model in self.models.items():
            model_path = os.path.join(save_path, f'{model_name}_model.pkl')
            # kaydet
            joblib.dump(
                value = model,
                filename= model_path
            )
            logger.info(f"{model_name} Modeli {save_path} Dosya Yoluna Kaydedildi")
            
            
            # oznitelikleri kaydet
            features_info = {
                'feature_names': list(self.X_train_processed.columns),
                'n_features': len(self.X_train_processed.shape[1]),
                'preprocessing_config': self.config.get('preprocessing', {})
            }
            # features_info'yu save_path'e kaydet
            joblib.dump(
                value = features_info,
                filename = os.path.join(save_path,'features_info.pkl') 
            )
            logger.info(f"Model Oznitelikleri Kaydedildi")
            
    def load_models(self, load_path = 'models/'):
        """Modelleri models klasorune yukler"""
        try:
            # preprocesser nesnesi yuklu mu
            self.preprocessor = joblib.load(
                filename= os.path.join(load_path, 'preprocessor.pkl')
            )
            logger.info("On Isleme Nesnesi Olusturuldu")
            
            # modelleri yukle
            model_files = [file for file in os.listdir(load_path) if file.endswith('_model.pkl')] 
            for model_file in model_files:
                model_name = model_file.replace('_model.pkl', '')
                model_path = os.path.join(load_path, model_file)
                self.models[model_name] = joblib.load(model_path)
                logger.info(f"{model_name} Modeli Yuklendi")
                
            feature_info = joblib.load(os.path.join(
                load_path,
                'feature_info.pkl'
            ))
            logger.info(f"Model Oznitelikleri Yuklendi. {feature_info['n_features']} Adet Oznitelik Bulunmaktadir.")
            logger.info(f"Modeller Yuklendi. Dosya Yolu: {load_path}")
        
        except Exception as err:
            logger.error(f"Model Yuklemesinde Beklenmeyen Bir Hata Olustu: {err}")
    
    
    def predict(self, data, model_name = "random_forest"):
        """Varsayilan model olarak RandomForestClassifier() kullanir ve tahmin yapar"""
        # gecerli bir model mi?
        if model_name not in self.models:
            logger.error(f"Gecersiz Model Ismi {model_name} Girdiniz: ")
            return None, None
        if not hasattr(self, 'preprocessor') or self.preprocessor is None:
            logger.error(f"Lutfen On Isleme Nesnesini (preprocessor) Bos Birakmayiniz")
            return None, None
            
            
        # islenmis veri (data_processed) ile calis
        try:
            data_processed = self.preprocessor.transform_model(data)
        except Exception as err:
            logger.error(f"Veri On Isleme Asamasinda Beklenmedik Hata Olustu: {err}")
            return None, None 
        
        # model
        model = self.models[model_name]
        
        if model_name in ['isolation_forest', 'lof']:
            # denetimsiz ogrenme (kumeleme gib)
            predictions = model.predict(data_processed)
            probabilities = np.where(predictions == -1, 0.8, 0.2) # aykırı deger donusumu yapiyoruz
            logger.info(f"{model_name} Modeli Kullanilarak Denetimsiz Ogrenme Tahmin Islemi Gerceklestirildi")
            return predictions, probabilities

        try:
            # denetimli ogrenme
            predictions = model.predict(data_processed)
            probabilities = model.predict_proba(data_processed)[:, 1] # pozitifleri al
            logger.info(f"{model_name} Modeli Kullanilarak Denetimli Ogrenme Tahmin Islemi Gerceklestirildi")
            return predictions, probabilities
        except Exception as err:
            logger.error(f"{model_name} Modeliyle Tahmin Yapilirken Bilinmeyen Hata Olustu: {err}")
            return None, None
        
        
    def _find_best_model(self):
        """ROC-AUC degerine gore en iyi modeli ve skorunu bulur"""
        best_model_name = None 
        best_model_roc_auc_score = 0
            
        # dongu ile don
        for model_name, evaluater in self.evaluators.items():
            if evaluater.results and 'roc_auc' in evaluater.results:
                roc_auc = evaluater.results['roc_auc']
                if best_model_roc_auc_score < roc_auc:
                    best_model_name = model_name
                    best_model_roc_auc_score = roc_auc
        
        if best_model_name is None:
            logger.warning(f"'best_model_name,' argumani None Olamaz")
            best_model_name = 'random_forest'
        
        logger.info(f"En Iyi Model:{best_model_name}\nEn Iyi ROC_AUC Skoru: {best_model_roc_auc_score}")
        return best_model_name # or 'random_forest' best_model_name = 'random_forest' ekledigimiz icin gerek kalmadi
    
    
    def run_pipeline(self, data_path = None, save_models = True, use_kagglehub = False):
        """Veriyi yukler, on isleme adimlarindan gecirir, modeli egitir, modeli degerlendirir"""
        logger.info("Veri Bilimi Adimlariniz Adim Adim Baslatiliyor")
        
        try:
            # 1.Adim => Veriyi Yuklemek
            self.load_fraud_data_from_kagglehub(
                data_path = data_path,
                download_with_kagglehub = use_kagglehub,
                synthetic = False
            )
            
            # 2. Adim => Verinin On Islenmesi
            self.preprocess_data()
            
            # 3. Adim => Verinin Modellenmesi => Modelin Egitilmesi
            self.train_models()
            
            
            # 4. Adim Modelin Degerlendirilmesi
            self.evaluate_models()
            
            # 5. Sonuclarin aciklanabilirligini gozlemle
            self.explain_model()
            
            # 6. En iyi modeli getir
            best_model = self._find_best_model()
            
            # 7. En iyi modelin aciklanabilirligini goster
            self.explain_model(model_name = best_model)
            
            # save_models parametresi True ise models/ klasoru altina kaydet
            if save_models:
                self.save_models() # save_path parametresi
                logger.info(f"{best_model} Modelinizin Bilgileri Kaydediliyor...")
                
            logger.info(f"Veri Bilimi Adimlariniz Sirasiyla Gerceklestirildi")
            
        except Exception as err:
            logger.error(f"Veri Bilimleri Adimlarinda Beklenmeyen Hata Olustu: {err}")
            return False 



def main():
    """ Command-Line Interface (Komut Satiri Arayuzu) """
    
    # argumanlari tanimlayalim
    parser = argparse.ArgumentParser(description= 'Fraud Detection Pipeline')
    
    parser.add_argument("--config", default = 'config/configuration.yaml', help = 'Configuration Path (Yapilandirma Dosya Yolu)')
    
    # veri dosyasini ekle
    parser.add_argument("--data", help = "Data File Path (Veri Dosya Yolu)")
    
    # egitim, test, tahmin olarak mode belirle
    parser.add_argument("--mode", choices = ['train', 'predict', 'explain'], default = 'train')
    
    # model belirle
    parser.add_argument("--model", default= 'random_forest', help = "Model Name For Prediction(Tahminleme Icin Model ismi) ")
    
    # Modelleri yukle
    parser.add_argument("--load_models", action = 'store_true', help = "Load Existing Models (Var Olan Modelleri Yukler)")
    
    # Modelleri kaydet
    parser.add_argument("--save_models", action = 'store_true', help = "Save Trained Models (Egitilmis Modelelri Kaydeder)")
    
    # Veri seti nereden yuklenecek?
    parser.add_argument("--use_kagglehub", action = 'store_true', help = 'Download Data With KaggleHub (KaggleHub ile Veri Setini Yukler)')
    
    # arguman listesinde tutalim
    args = parser.parse_args()
    
    # Olusturulan PipeLine'i calistir
    fraud_pipeline = FraudDetectionPipeline(args.config)
    
    # Yukarida belirtilen egitim (train), aciklama (explain) ve tahmin (predict) modunu tanimlayalim
    if args.mode == 'train':
        success = fraud_pipeline.run_pipeline(
            data_path = args.data,
            save_models = args.save_models,
            use_kagglehub = args.use_kagglehub     
        )
        # Cikis yap
        sys.exit(0 if success else 1) #basariliysa 0, basarisizda 1 
    
    elif args.mode == 'explain':
        if args.load_models():
            fraud_pipeline.load_models()
        
        #Veri setini yukle    
        fraud_pipeline.load_fraud_data_from_kagglehub(synthetic= False)
        
        # Veri setini on islemeden gecir
        fraud_pipeline.preprocess_data()
        
        # Onem derecesi ve anomali oruntulerini ata
        importance, pattern = fraud_pipeline.explain_model(model_name = args.model)
        
        print(f"En Onemli 10 Degisken (Top 10 Important Features)")
        for idx, (feature, score) in enumerate(list(importance.items())[-10:]):
            print(f"{idx + 1}. {feature} Degiskeninin Onem Skoru: {score:.4f}")
    
    elif args.mode == 'predict':
        # egitilmis modelleri yukle
        if args.load_models:
            fraud_pipeline.load_models()
            
        # Veriyi yukle ve onu on isleme adimlarina sok
        fraud_pipeline.load_fraud_data_from_kagglehub(synthetic=False)
        fraud_pipeline.preprocess_data()
        
        # tahmin degerlerini ve olasiklarini al
        predictions, probabilities = fraud_pipeline.predict(
            data = fraud_pipeline.X_test_processed.head(),
            model_name = args.model
        )
        
        # 5 adet gozlemin tahmin degerini ve olasiligini alalim
        for idx, (prediction, probability) in enumerate(zip(predictions[:5], probabilities[:5])):
            print(f"Ornek_{idx} Tahmin Degeri: {prediction}; Olasiligi: {probability:.2f}")
            
        
        print("Orneklemler Uzerinden Tahminleme")


if __name__ == '__main__':
    main()