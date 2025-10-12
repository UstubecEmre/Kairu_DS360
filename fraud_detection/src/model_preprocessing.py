#%% import required libraries (Gerekli Kutuphaneleri Dahil Et)
from pathlib import Path
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

# for model preprocessing (model on isleme icin)
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder # For encoding
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler # for scaling
from sklearn.model_selection import train_test_split # for train and test split
from sklearn.impute import SimpleImputer, KNNImputer # for filling na values (Eksik verileri doldurmak icin)


# for imbalanced data set
from imblearn.over_sampling import SMOTE, ADASYN 
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek 


import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings('ignore')



class Feature_Preprocessor:
    """Dolandiricilik tespiti icin kullanilir"""
    # constructor method
    def __init__(self, scaling_method = 'robust', encoding_method = 'onehot'):
        
        self.scaling_method = scaling_method
        self.encoding_method = encoding_method
        
        # encoding method
        if encoding_method == 'label':
            self.encoder = LabelEncoder()
            
        elif encoding_method == 'ordinal':
            self.encoder = OrdinalEncoder(handle_unknown= 'use_encoded_value', unknown_value = -1)
        
        elif encoding_method == 'onehot':
            self.encoder = OneHotEncoder(handle_unknown = 'ignore', sparse_output = False)

        else:
            raise ValueError("Lutfen gecerli bir encoding yontemi giriniz: (onehot, label, ordinal)")
        
        if scaling_method == 'standart':
            self.scaler = StandardScaler()
        elif scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaling_method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError("Lutfen gecerli bir scaling yontemi seciniz. (standart, minmax, robust)")
        
        # define empty list for features (degiskenler icin bos liste tanimla) attributes placeholder
        self.categorical_features = []
        self.numerical_features = []
        self.is_fitted = False 
        self.encoded_features = [] # for categorical features (Kategorikler icin)
        
    def identify_real_data_types(self, dataframe):
        """Ozniteliklerin gercek veri tipini belirler."""
        # for categorical features (Kategorik veriler icin)
        self.categorical_features = dataframe.select_dtypes(includes = ['object', 'category']).columns.tolist()
            
        # for numerical features (sayisal veriler icin)
        self.numerical_features = dataframe.select_dtypes(includes = ['float64', 'int64']).columns.tolist()
            
        logger.info(f"Kategorik Degisken Sayisi: {len(self.categorical_features)}")
        logger.info(f"Sayisal Degisken Sayisi: {len(self.numerical_features)}")
        
    def handle_missing_values(self, 
                              dataframe:pd.DataFrame,
                              numerical_strategy: str = 'median',
                              categorical_strategy: str = 'most_frequent') -> pd.DataFrame:
        
        self.dataframe = dataframe.copy() 
        df_processed = dataframe.copy()
        
        
        # build a Pipeline => ColumnTransformer
        
        # for numerical missing values (eksik sayisal degerler icin doldurma)
        if hasattr(self, 'numerical_features') and self.numerical_features:
            try:
                num_imputer = SimpleImputer(strategy = numerical_strategy)
                df_processed[self.numerical_features] = num_imputer.fit_transform(
                    df_processed[self.numerical_features]
                )
            except Exception as err:
                logger.error(f"Beklenmeyen Bir Hata Olustu. Sayisal Degiskenler Doldurulamadi: {err}")
                return False 
        
        # to fill for categorical features (kategorik degiskenleri doldurmak icin)
        if hasattr(self, 'categorical_features') and self.categorical_features:    
            try:
                cat_imputer = SimpleImputer(strategy= categorical_strategy)
                df_processed[self.categorical_features] = cat_imputer.fit_transform(
                    df_processed[self.categorical_features]
                )
            except Exception as err:
                logger.error("Sayisal Degiskenler Medyan Degerleriyle; Kategorik Degiskenler Mod Ile Dolduruldu")
                return False
        logger.info(f"Sayisal Degiskenler {numerical_strategy} ile dolduruldu")
        logger.info(f"Kategorik Degiskenler {categorical_strategy} ile dolduruldu")
        
        return df_processed
    
    def detect_outlier_values(self, 
                              dataframe: pd.DataFrame,
                              method: str =  'iqr', 
                              threshold: float = 1.5) -> dict:
        """Aykiri degerleri tespit eder. IQR ve ZScore yontemleri kullanilir. 
        Args:
            df (pd.DataFrame): Giris DataFrame'i 
            method: Aykiri deger tespiti icin kullanilacak yontem (iqr, zscore)
            threshold (float): Esik degeri
        
        Returns:
            outlier_info (dict)
        """
        outlier_info = {}
        self.dataframe = dataframe.copy()
        
        if hasattr(self, 'numerical_features') and self.numerical_features:
            try:
                for col in self.numerical_features:
                    if method == 'îqr':
                        Q1 = dataframe[col].quantile(0.25)
                        Q3 = dataframe[col].quantile(0.75)
                        IQR = Q3 - Q1 
                        upper_bound = Q3 + threshold * IQR 
                        lower_bound = Q1 - threshold * IQR 
                        
                        outliers = dataframe[(dataframe[col] > upper_bound) | (dataframe[col] < lower_bound)]
                        outlier_info[col] = {
                            "count": len(outliers)
                            ,"upper_bound": upper_bound
                            ,"lower_bound": lower_bound
                            ,"percentage": (len(outliers) / len(dataframe)) * 100
                        }
                    elif method == "zscore":
                        """Sutunun ortalama degerini cikar ve standart sapmaya bol"""
                        z_scores = np.abs((dataframe[col] - dataframe[col].mean()) / dataframe[col].std())
                        outliers = dataframe[z_scores > threshold]
                        outlier_info[col] = {
                            "count": len(outliers)
                            ,"threshold": threshold
                            ,"percentage": (len(outliers) / len(dataframe)) * 100
                        }
            except Exception as err:
                logger.error(f"Beklenmedik Hata Olustu: {err}")
                return {} 
        return outlier_info
    
    
    
    def extract_new_features(self, dataframe: pd.DataFrame)->pd.DataFrame:
        """Ozellik muhendisligi gerceklestirir, yeni ozellikler turetir
        Args:
            dataframe (pd.DataFrame): Girdi DataFrame'i
        
        Returns:
            df_featured (pd.DataFrame): Ozellik muhendisligi gerceklestirilmis DataFrame
        
        """
        df_featured = dataframe.copy()
        
        # Carpikligi olan degiskenlere log1p donusumu uygula
        if 'Amount' in df_featured.columns:
            if (df_featured['Amount'] < 0).any():
                logger.warning(f"'Amount' sutununda negatif degerler bulundu. log donusumu gerceklestirilemedi")
            else:
                df_featured['Amount_log_transformed'] = np.log1p(df_featured['Amount'])
            
        # Tarih-saat-zaman degiskenlerinden yeni ozellik turet
        if 'Time' in df_featured.columns:
            df_featured['Day_of_week'] = (df_featured['Time'] // (3600 * 24)) % 7 # bir hafta 7 gun, 1 saat 3600 saniye
            df_featured['Hour'] = (df_featured['Time'] // (3600)) % 24 # bir gun 24 saat
        
        # interaction features (ilk 2 sayisal degisken)
        if hasattr(self, 'numerical_features') and len(self.numerical_features) >= 2:
            feature_1, feature_2 = self.numerical_features[:2] # 0 ve 1. index degerleri sirasiyla ilgili degiskenlere atanir
            df_featured[feature_2] = pd.to_numeric(df_featured[feature_2], errors='coerce')
            # sifira bolum hatasini engelle 
            df_featured[f'{feature_1}_{feature_2}_ratio'] = df_featured[feature_1] / df_featured[feature_2].replace(0, np.nan)
            df_featured[f'{feature_1}_{feature_2}_multiplied'] = df_featured[feature_1] * df_featured[feature_2]
        
        logger.info("Ozellik Muhendisligi Gerceklestirildi")
        return df_featured
    
    def fit_transform_model(self, dataframe:pd.DataFrame, target_col:str=None):
        """Modeli egitir ve donusturur.
        
        Args:
            dataframe (pd.DataFrame): Girdi DataFrame'i
            target_col (str, optional): Hedef degisken. Defaults to None.

        Returns:
            processed_df (pd.DataFrame): Ozellik muhendisligi, olcekleme ve kategorik degiskenlerin sayisala donusumu gerceklestirilmis DataFrame
            target (str): Hedef degisken ismi
        """
        
        df_processed = dataframe.copy()
        
        # Hedef degisken y olarak atanmali, X'de yer almamali
        if isinstance(target_col, str) and target_col in df_processed:
            target = df_processed[target_col]
            df_processed = df_processed.drop(columns = [target_col])         
        else:
            target = None
        # Veri tiplerini belirle
        self.identify_real_data_types(df_processed)
            
        # Eksik degerleri doldur (impute)
        df_processed = self.handle_missing_values(df_processed)
            
        # Ozellik muhendisligi gerceklestir
        df_processed = self.extract_new_features(df_processed)
            
        # Yeni degerlerin veri tipini belirle
        self.identify_real_data_types(df_processed)
            
        # Sayisal degiskenleri olceklendir; kategorikleri sayisala cevir
        if hasattr(self, 'numerical_features') and self.numerical_features:
            try:
                df_processed[self.numerical_features] = self.scaler.fit_transform(
                    df_processed[self.numerical_features]
                )
                logger.info(f"Kullanilan Olceklendirme Yontemi: {self.scaling_method}")
            except Exception as err:
                logger.error(f"Beklenmedik Bir Hata Olustu. Sayisal Degiskenler Olceklendirilemedi: {err}")
        
        # Kategorik degiskenleri sayisala donustur    
        if hasattr(self, 'categorical_features') and self.categorical_features:
            try:
                if isinstance(self.encoding_method, str) and self.encoding_method == 'onehot':
                    encoded_data = self.encoder.fit_transform(
                        df_processed[self.categorical_features]
                    )
                        
                    # degisken isimlerini alalim 
                    if hasattr(self.encoder, 'get_feature_names_out'):
                        self.encoded_feature_names = self.encoder.get_feature_names_out(
                            self.categorical_features
                        ).tolist()
                    else:
                        self.encoded_feature_names = [
                            f"{cat}_{val}" for cat in self.categorical_features
                            for val in self.encoder.categories_[
                                self.categorical_features.index(cat)
                            ]
                        ]
                            
                # Sayisala donusturulmus bir DataFrame olustur
                    encoded_df = pd.DataFrame(
                        encoded_data,
                        columns = self.encoded_feature_names,
                        index = df_processed.index
                        )
                        
                    df_processed = df_processed.drop(columns= self.categorical_features)
                    df_processed = pd.concat([df_processed, encoded_df], axis = 1)
                else:
                     for col in self.categorical_features:
                         df_processed[col] = self.encoder.fit_transform(
                             df_processed[col]
                         )   
            except Exception as err:
                logger.error(f"Beklenmeyen Bir Hata Olustu. Encoding Islemi Gerceklestirilemedi: {err}")
            
            
        self.is_fitted = True 
            
        # hedef degiskeni ekle
        if target_col:
            df_processed[target_col] = target
                
        return df_processed, target 


    def transform_model(self, dataframe: pd.DataFrame, target_col:str=None):
        """Egitilmis model uzerinde donusturme islemleri gerceklestirir

        Args:
            dataframe (pd.DataFrame): _
            target_col (str, optional): Hedef degisken. Defaults to None.

        Raises:
            ValueError: Uygun veri tipi kullanilmamasi durumunda deger hatasi dondurur

        Returns:
            processed_df (pd.DataFrame): Transform edilmis DataFrame'i dondurur.
            target (str): Hedef degikeni dondurur
        """
        if not self.is_fitted:
            raise ValueError("Fit Transform Uygulamadan Sadece Transform Calistiramazsiniz!!!")
        
        df_processed = dataframe.copy()
        target = None 
        if isinstance(target_col, str) and target_col in df_processed.columns:
            target = df_processed[target_col]
            df_processed = df_processed.drop(columns= [target_col])
    
        # Eksik deger varsa doldur
        df_processed = self.handle_missing_values(df_processed)
        
        # Oznitelik uret
        df_processed = self.extract_new_features(df_processed)
        
        # scaling and encoding
        if hasattr(self, "numerical_features") and self.numerical_features:
            try:
                # var olan degerleri donustur
                existing_numerical_features = [num_feat for num_feat in self.numerical_features if num_feat in df_processed.columns]
            
                if existing_numerical_features:
                    df_processed[existing_numerical_features] = self.scaler.transform(
                        df_processed[existing_numerical_features]
                    )
            except Exception as err:
                logger.error(f"Beklenmeyen Bir Hata Olustu: {err}")
        
        if hasattr(self, "categorical_features") and self.categorical_features:
            try:
                existing_categorical_features = [cat_feat for cat_feat in self.categorical_features
                                                 if cat_feat in df_processed.columns]
                
                if existing_categorical_features and self.encoding_method == 'onehot':
                   encoded_data = self.encoder.transform(
                        df_processed[existing_categorical_features]
                    )
                   encoded_df = pd.DataFrame(
                       encoded_data
                       ,columns = self.encoded_feature_names
                       ,index = df_processed.index
                        
                   )   
                   df_processed = df_processed.drop(columns = existing_categorical_features)
                   df_processed = pd.concat([df_processed, encoded_df], axis = 1) 
                
                elif existing_categorical_features:
                    for cat_col in existing_categorical_features:
                        df_processed[cat_col] = self.encoder.transform(df_processed[cat_col])
                
                else:
                    logger.warning("Gecersiz Bir Islem Gerceklestirdiniz")
            except Exception as err:
                logger.error(f"Beklenmeyen Bir Hata Olustu: {err}")
                
        if target_col and target is not None:
            df_processed[target_col] = target
            
        return df_processed, target


    def show_distributions(self, df_original:pd.DataFrame, df_processed:pd.DataFrame, target_col:str = None):
        if not hasattr(self, "numerical_features"):
            logger.warning("'numerical_features' isminde bir ozellik bulunmamaktadir")
            return False 
            
            
        try:
            num_cols = [col for col in self.numerical_features
                        if col in df_original.columns]
            if not num_cols:
                logger.warning("Beklenmeyen Sutun Gozlemlendi")
                    
            
            n_cols = len(num_cols)
            fig, axs = plt.subplots(2, n_cols, figsize = (5 * n_cols, 8))
            axs = np.array(axs).reshape(2, n_cols)
                    
            for idx, col in enumerate(num_cols):
                axs[0, idx].hist(df_original[col], bins = 100, alpha = 0.8, color = "mediumpurple")
                axs[0, idx].set_title(f"Original_{col} Sutunu")
                axs[0, idx].set_xlabel(f"{col}")
                axs[0, idx].set_ylabel("Frekans")
                        
                if col in df_processed.columns:
                    axs[1, idx].hist(df_processed[col], bins = 100, alpha = 0.8, color = "orange")
                    axs[1, idx].set_title(f"Processed_{col} Sutunu")
                    axs[1, idx].set_xlabel(f"{col}")
                    axs[1, idx].set_ylabel("Frekans")
            plt.tight_layout()
            plt.grid(axis= "y")
            plt.show()
                
        except Exception as err:
            logger.error(f"Beklenmeyen Bir Hata Olustu: {err}")
             
    def get_feature_informations(self, dataframe:pd.DataFrame)->dict:
        """Degiskenlerin ozelliklerini dondurur"""
        feature_information = {
            "categorical_features": self.categorical_features
            ,"scaling_method": self.scaling_method
            ,"numerical_features": self.numerical_features
            ,"encoding_method": self.encoding_method
            ,"encoded_feature_names": self.encoded_features
        }
        return feature_information
    


class ImbalanceHandler:
    """Dengesiz veri setlerinde SMOTE, ADASYN, SMOTETOMEK yontemlerini uygular"""
    @staticmethod
    def find_target_class_ratio(y, target_cols = None):
        unique_val, nunique_val = np.unique(y, return_counts=True)
        
        if len(nunique_val) == 0:
            logger.warning("Hedef Degisken Bos!!! Dengesizlik Orani Hesaplanamadi")
            return False 
        
        print("Sinif Dagilimi")
        for idx, (cls, count) in enumerate(zip(unique_val, nunique_val)):
            
            if target_cols and idx < len(target_cols):
                col = target_cols[idx]
            else:
                col = f"class {cls}"
            
            percentage = (count / len(y)) * 100
            print(f"{col} sutunu: {count} Orani: {percentage:.4f}%")
        
        # dengesizlik orani
        imbalance_ratio = max(nunique_val) / min(nunique_val)
        print(f"Veri Setinin Dengesizlik Orani: {imbalance_ratio:.4f}")

        # gorsellestirme notebooks icerisinde var. Dilerseniz buraya da ekleyebilirsiniz
        
        result_dict = {
            "class_counts": dict(zip(unique_val, nunique_val))
            ,"imbalance_ratio": imbalance_ratio
        }
        return result_dict 
        
        
    @staticmethod
    def apply_smote(X, y, sampling_strategy = 'auto', random_state = 42):
        smote = SMOTE(sampling_strategy= sampling_strategy, random_state= random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        logger.info(f"Sentetik Veri Uretimi Gerceklestirildi. {len(X)} -> {len(X_resampled)}")
        return X_resampled, y_resampled
    
    @staticmethod
    def apply_adasyn(X, y, sampling_strategy = 'auto', random_state = 42):
        adasyn = ADASYN(sampling_strategy = sampling_strategy, random_state = 42)
        X_resampled, y_resampled = adasyn.fit_resample(X,y)
        logger.info(f"ADASYN Methodu Uygulandi: {len(X)} -> {len(X_resampled)}")
        return X_resampled, y_resampled
    
    @staticmethod
    def apply_smotetomek(X, y, sampling_strategy = 'auto', random_state = 42):
        smotemek = SMOTETomek(sampling_strategy = sampling_strategy, random_state= random_state)
        X_resampled, y_resampled = smotemek.fit_resample(X,y)
        logger.info(f"SMOTETomek Methodu Uygulandi: {len(X)} -> {len(X_resampled)}")
        return X_resampled, y_resampled


def demo_preprocessing(data_path =Path(r"D:\Kairu_DS360_Projects\fourth_week_project\fraud_detection\data\anomaly_scores_raw.csv")):
    """Islenmis egitim ve test veri setini kaydeder, opsiyonel olarak dengesiz veri setini dengeli hale getirir """
    try:
        fraud_df = pd.read_csv(data_path)
        print(fraud_df.head())
        assert "Class" in fraud_df.columns, "Hedef Degisken 'Class' Bulunamadi"
    except FileNotFoundError:
        raise FileNotFoundError("Dosya Yolu Bulunamadi")
    except Exception as err:
        logger.error(f"Beklenmeyen Bir Hata Olustu: {err}")
        
    print("Ham Veri Seti Bilgileri:\n")
    fraud_df.info()
    
    
    # Dengesiz Veri Seti mi?
    print("Veri Seti Hedef Degisken ('Class') Dagilimi")
    ImbalanceHandler.find_target_class_ratio(fraud_df['Class'], ['Normal','Fraud'])
    
    
    # Veri egitim ve test diye ayrilmis mi?
    if 'split' in fraud_df.columns:
        # split varsa
        feature_cols = [col for col in fraud_df.columns if col not in('Class', 'split')]
        train_fraud_df = fraud_df[fraud_df["split"] == 'train'].reset_index(drop = True)
        test_fraud_df = fraud_df[fraud_df['split'] == 'test'].reset_index(drop = True)
        
        
        X_train = train_fraud_df[feature_cols].copy()
        y_train = train_fraud_df['Class'].astype(int).copy()
        
        X_test = test_fraud_df[feature_cols].copy()
        y_test = test_fraud_df['Class'].astype(int).copy()
    # split sutunu yoksa
    else:
        X = fraud_df.drop(columns = ['Class'], errors = 'ignore')
        y = fraud_df['Class'].astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, stratify= y)
        
    # encoding ve scaling uygula, modeli egitime hazir hale getir
    preprocessor = Feature_Preprocessor(scaling_method= 'robust', encoding_method= 'onehot')
    
    # X_train_processed
    X_train_processed = preprocessor.fit_transform_model(
        pd.concat([X_train, y_train.rename('Class')], axis = 1), target_col = "Class"
    )
    y_train_processed = X_train_processed['Class'].astype(int)
    X_train_processed = X_train_processed.drop('Class', axis=1)
    
    # transform 
    X_test_processed = preprocessor.transform_model(
        pd.concat([X_test,  y_test.rename('Class')], axis = 1)
        ,target_col = 'Class'
    )
    y_test_processed = X_test_processed['Class'].astype(int)
    X_test_processed = X_test_processed.drop(columns = 'Class', axis = 1)
    
    print(f"Islenmis Veri Setinin Egitim Seti Boyutu: {X_train_processed.shape}")
    print(f"Islenmis Veri Setinin Test Seti Boyutu: {X_test_processed.shape}")
    
    
    # Dengesiz veriyi dengeli hale getirelim
    try:
        X_train_balanced,  y_train_balanced = ImbalanceHandler.apply_smote(
            X_train_processed
            ,y_train_processed
        )
    except Exception as err:
        logger.warning(f"Sentetik Veri Uretiminde Beklenmeyen Bir Hata Olustu: {err}")
        X_train_balanced, y_train_balanced = X_train_processed, y_train_processed
        
    # Dosyalari kaydedelim
    from pathlib import Path
    DATA_DIRECTORY = Path("./data/processed")
    DATA_DIRECTORY.mkdir(parents = True, exist_ok= True)
    # dosya yollarini belirt
    fraud_train_out = DATA_DIRECTORY / "train_processed_supervised.csv"
    fraud_test_out  = DATA_DIRECTORY / "test_processed_supervised.csv"
    fraud_full_out  = DATA_DIRECTORY / "dataset_processed_supervised.csv"
    
    # fraud_train_out csv formatinda kaydet
    pd.concat([X_train_processed.reset_index(drop = True), y_train_processed.reset_index(drop = True)], axis = 1).to_csv(fraud_train_out, index = False)
    
    # fraud_test_out csv formatinda kaydet
    pd.concat([X_test_processed.reset_index(drop = True), y_test_processed.reset_index(drop = True)], axis = 1).to_csv(fraud_test_out, index = False)
    
    # split etiketi varsa:
    if 'split' in fraud_df.columns:
        train_merge = pd.concat([
            X_train_processed.reset_index(drop = True)
            ,y_train_processed.reset_index(drop = True)
            ,pd.Series(['train'] * len(y_train_processed), name = 'split')
        ]
        ,axis = 1)
        
        test_merge = pd.concat([
            X_test_processed.reset_index(drop = True)
            ,y_test_processed.reset_index(drop = True)
            ,pd.Series(['test'] * len(y_test_processed), name = 'split')
        ]
        ,axis = 1)
        
        
        full_fraud_df = pd.concat([train_merge, test_merge], axis = 0, ignore_index= True)
    
    else:
        full_fraud_df = pd.concat(
            [
                pd.concat([
                    X_train_processed.reset_index(drop = True)
                    ,y_train_processed.reset_index(drop = True)
                ],axis = 1)
                
                ,pd.concat([
                    X_test_processed.reset_index(drop = True)
                    ,y_test_processed.reset_index(drop = True)
                ], axis = 1)
            ]
            ,axis = 0
            ,ignore_index= True
        )
    # full_fraud_df csv formatinda kaydet
    full_fraud_df.to_csv(fraud_full_out, index = False)
        
    print(f"Egitim Cikti Dosyasi Kaydedildi (CSV): {fraud_train_out}")
    print(f"Test Cikti Dosyasi Kaydedildi (CSV): {fraud_test_out}")
    print(f"Train Test Birlesik Cikti Dosyasi Kaydedildi (CSV): {fraud_full_out}")
        
    result_dict = {
            'X_train': X_train_processed
            ,'X_test': X_test_processed
            ,"y_train": y_train_processed
            ,"y_test": y_test_processed
            ,"X_train_balanced": X_train_balanced
            ,"y_train_balanced" : y_train_balanced
    }
    return result_dict    
    
if __name__ == '__main__':
    demo_preprocessing() 
    