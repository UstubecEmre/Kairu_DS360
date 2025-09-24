#%% import required libraries
import os # for operating system operations (like making directory etc...)
# import numpy as np # for linear algebra, convert to data type
import pandas as pd # for data manipulation
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


#%% define clean_iris_data(input_path, output_path)
def clean_iris_data(input_path = 'data/raw/iris.csv', output_path = 'data/processed/iris_processed.csv'):
    """ Cleans, encodes and applies feature engineering to the Iris dataset."""
    try:
    # load dataset
        iris_df = pd.read_csv(input_path)
    except FileNotFoundError:
        raise FileNotFoundError(f'Error: The file at {input_path} not found!!!')
        # return None, None 
    print("Starting data encoding and feature engineering:)")
    
    
    # make a copy to avoid modifying the original DataFrame
    iris_cleaned = iris_df.copy()
    
    # There are not any missing values
    
    # drop 'id' column
    if 'Id' in iris_cleaned.columns:
        iris_cleaned = iris_cleaned.drop('Id', axis = 1)
        print("Dropped 'Id' column...")
    
    # apply one hot encoding to the categorical column
    
    #iris_encoded = pd.get_dummies(iris_cleaned, columns=['species'], drop_first= True)
    #print("Applied OHE to 'species' column...")
    le = LabelEncoder()
    iris_cleaned['species_encoded'] = le.fit_transform(iris_cleaned['species'])
    # use feature extraction techniques
    
    iris_cleaned['sepal_ratio'] = iris_cleaned['sepal_length'] / iris_cleaned['sepal_width']
    iris_cleaned['petal_ratio'] = iris_cleaned['petal_length'] / iris_cleaned['petal_width']
    print("Created new features => 'sepal_ratio' and 'petal_ratio'")
    
    
    # make output path
    os.makedirs(os.path.dirname(output_path), exist_ok= True)
    
    # save cleaned and encoded dataset
    iris_cleaned.to_csv(output_path, index = False)
    
    # show some information about cleaned and encoded dataset
    print(f"Encoded dataset saved: {output_path}")
    print(f"Total Missing Values: \nThere are {iris_cleaned.isnull().sum().sum()} missing values")
    
    features = list(iris_cleaned.columns)
    
    print(f"Model features: {features}")
    
    # return cleaned and encoded dataset and new features
    return iris_cleaned, features


#%% call the function
if __name__ == '__main__':
    clean_iris_data()
    