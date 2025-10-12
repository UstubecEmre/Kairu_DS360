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
    
# %%
