#%% import required libraries
import os # for operating system
import pandas as pd # for data manipulation
import seaborn as sns # for dataset loading 

#%% define download_iris() function
def download_iris_data():
    """Load iris dataset from seaborn"""
    
    # create directories
    os.makedirs('data/raw',exist_ok= True)
    
    # load iris data set
    iris_df = sns.load_dataset('iris')
    
    # save the raw data
    iris_df.to_csv('data/raw/iris.csv', index = False)
    
    # show basic information about iris dataset
    print("Iris dataset loaded successfully:)")
    print(f"Iris dataset shape: {iris_df.shape}")
    print(f"Iris dataset total record: {iris_df.shape[0]}")
    print(f"Columns of Iris dataset: {list(iris_df.columns)}")
    print(f"Total Missing Values of Iris Dataset:\n {iris_df.isnull().sum()}")
    
    
    return iris_df 


#%% call the download_iris_data on main function
if __name__== '__main__':
    # call the download_iris_data()
    download_iris_data()