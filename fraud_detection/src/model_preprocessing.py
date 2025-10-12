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
        