#%% import required libraries (Gerekli kutuphaneleri yukle)

# for operation system and directory operations (isletim sistemi ve dosya islemleri)
import os
from pathlib import Path
import json
# for data manipulation and visualization (veri analizi ve gorsellestirme)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 


# for modelling and model evaluation
from sklearn.ensemble import  IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve



#%% Islenmis veri klasoru olustur
RAW_DIR = Path(r"D:\Kairu_DS360_Projects\fourth_week_project\fraud_detection\data\raw\creditcard_fraud.csv")
assert RAW_DIR.exists(), f"Ham Veri Dosya Yolu Bulunamadi: {RAW_DIR}"
# fraud_df = pd.read_csv(RAW_DIR)


try:
    PROJECT_DIR = Path(__file__).resolve().parents[1] # fraud detection klasoru altinda olussun, her cagrildiginda ilgili klasorde olusmasin.
except NameError:
    PROCESSED_DIR = Path.cwd()
except Exception as err:
    raise Exception(f"Beklenmeyen Bir Hata Olustu: {err}")


PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

def ensure_processed_dir():
    try:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        print("Islenmis Veri Klasoru Olusturuldu")
    except FileNotFoundError:
        raise FileNotFoundError(f"Ilgili Klasor Olusturulamadi {PROCESSED_DIR}")
    except Exception as err:
        raise Exception(f"Beklenmeyen Bir Hata Olustu: {err}")
    
OUTLIERS_CSV = PROCESSED_DIR / "anomaly_scores_raw.csv"
OUTLIER_META_CSV = PROCESSED_DIR / "outlier_meta_raw.json"

#%% aykiri degerler icin esik deger belirle 
def detect_outlier_threshold_by_f1(y_true, scores):
    """ 
    Precision Recall Egrisini kullanarak Anomali Tespiti Gerceklestirir
    """
    
    prec, rec, threshold = precision_recall_curve(y_true, scores)
    
    # f1 hesapla (2 * recall * precision) / (recall + precision) 
    f1 = (2 * (prec * rec)) / (prec + rec + 1e-9)
    best_index = int(np.argmax(f1))
    threshold_choice = float(threshold[
        max(
            0,
            min(best_index - 1, len(threshold) - 1))
        ] 
        if len(threshold) > 0 else 0.0)
    
    return {
        "threshold": threshold_choice
        ,"precision": float(prec[best_index])
        ,"recall": float(rec[best_index])
        ,"f1": float(f1[best_index])
    }



def main():
    assert RAW_DIR.exists(), f"Ham Veri Bulunamadi : {RAW_DIR}"
    fraud_df = pd.read_csv(RAW_DIR)
    
    assert "Class" in fraud_df.columns, "Hedef Degisken 'Class' Bulunamadi"
    
    # strattified olusturalim
    if "split" not in fraud_df.columns:
        print("'split' Sutunu Bulunamadi. Simdi Olusturuluyor...")
        y_temp = fraud_df['Class'].astype(int).values 
        
        index_train, index_test = train_test_split(
            np.arange(len(fraud_df))
            ,test_size= 0.20
            ,stratify= y_temp
            ,random_state= 42
        )
        split = np.array(['train'] * len(fraud_df), dtype = object)
        split[index_test] = "test"
        fraud_df['split'] = split
        
        # feature cols => bagimsiz degiskenler
    feature_cols = [col for col in fraud_df.columns if col not in ('Class','split')]
    train_fraud = fraud_df[fraud_df['split'] == 'train'].reset_index(drop = True)
    test_fraud = fraud_df[fraud_df['split'] == 'test'].reset_index(drop = True)
        
        # bagimli ve bagimsiz degisken ayrimi
    X_train =  train_fraud[feature_cols].values
    X_test = test_fraud[feature_cols].values 
        
    y_test = test_fraud['Class'].astype(int).values
        
    print(f"Ham Veri Dosya Yolu: {RAW_DIR}")
    print(f"Egitim Veri Seti Boyutu: {X_train.shape}")
    print(f"Test Veri Seti Boyutu: {X_test.shape}")
    print(f"Test Veri Seti Dolandiricilik Orani: {y_test.mean():.6f}")
        
        
        
    # Isolation Forest modeli egit
    iso = IsolationForest(n_estimators= 500, random_state= 42, contamination = "auto")
    iso.fit(X_train)
    # Yuksek skorlar, anomoli yani aykiri degere karsilik gelmektedir.
    # Burada isaret degistirerek anomalileri alalim.
    iso_train_scores = -iso.decision_function(X_train)
    iso_test_scores = -iso.decision_function(X_test) 
        
    iso_threshold = detect_outlier_threshold_by_f1(y_true = y_test, scores = iso_test_scores)
    iso_test_alarm = (iso_test_scores >= iso_threshold['threshold']).astype(int)
        
        
    # roc and average precision skorlarini hesaplayalim
    iso_roc = float(roc_auc_score(y_true = y_test, y_score = iso_test_scores))
    iso_avg_prec = float(average_precision_score(y_true = y_test, y_score = iso_test_scores))
        
    iso_result_dict = {
        "roc_auc": iso_roc,
        "average_precision": iso_avg_prec,
        **iso_threshold,
        "alarm_rate": iso_test_alarm.mean()
        }
    
    # Local Outlier Factor => LOF
    # Burada da yuksek skor degeri anomali kabul edilir, bunlarin da ters isaretlisini almaliyiz
    lof = LocalOutlierFactor(n_neighbors= 30, contamination= "auto", novelty= True)
    lof.fit(X_train)
    lof_train_scores = -lof.score_samples(X = X_train)
    lof_test_scores = -lof.score_samples(X = X_test)
        
    lof_threshold = detect_outlier_threshold_by_f1(y_test, lof_test_scores)
    lof_test_alarm = (lof_test_scores >= lof_threshold['threshold']).astype(int)
    lof_roc = roc_auc_score(y_true = y_test, y_score = lof_test_scores) 
    lof_avg_prec = average_precision_score(y_true= y_test, y_score = lof_test_scores)
    """ 
    lof_result_dict = {
        "LOF_ROC_AUC": lof_roc
        ,"LOF_Average_Precision": lof_avg_prec
        ,"LOF_Threshold": lof_threshold['threshold']
        ,"LOF_Precision": lof_threshold['precision']
        ,"LOF_Recall": lof_threshold['recall']
        ,"LOF_F1_Score": lof_threshold['f1']
        ,"LOF_Threshold_Alarm_Rate": lof_test_alarm.mean()
            }
    """
    lof_result_dict = {
        "roc_auc": lof_roc,
        "average_precision": lof_avg_prec,
        **lof_threshold,
        "alarm_rate": lof_test_alarm.mean()
    }      
    
    fraud_out_df = fraud_df.copy()
    fraud_out_df['iso_score'] = np.nan 
    fraud_out_df['lof_score'] = np.nan 
    fraud_out_df['iso_alarm'] = 0
    fraud_out_df['lof_alarm'] = 0
    
    # iso dataframe
    fraud_out_df.loc[fraud_out_df['split'] == 'train', 'iso_score'] = iso_train_scores
    fraud_out_df.loc[fraud_out_df['split'] == 'test', 'iso_score'] = iso_test_scores
    fraud_out_df.loc[fraud_out_df['split'] == 'test', 'iso_alarm'] = iso_test_alarm
    
    # lof dataframe
    fraud_out_df.loc[fraud_out_df['split'] == 'train', 'lof_score'] = lof_train_scores
    fraud_out_df.loc[fraud_out_df['split'] == 'test', 'lof_score'] = lof_test_scores
    fraud_out_df.loc[fraud_out_df['split'] == 'test', 'lof_alarm'] = lof_test_alarm 
     
        
    # Kaydedelim
    fraud_out_df.to_csv(OUTLIERS_CSV, index = False)
    print(f"Aykiri Deger Tespit Dosyasi (CSV) Kaydedildi. Goz Atiniz => {OUTLIERS_CSV}")
    
    meta_data = {
    "input": {
        "file_path": str(RAW_DIR),
        "n_total": int(len(fraud_df)),
        "n_train": int(len(train_fraud)),
        "n_test": int(len(test_fraud)),
        "fraud_rate_test": float(y_test.mean())
    },
    "output": {
        "csv_path": str(OUTLIERS_CSV),
        "meta_path": str(OUTLIER_META_CSV)
    },
    "models": {
        "isolation_forest": {
            "parameters": {
                "n_estimators": 500,
                "contamination": "auto",
                "random_state": 42
            },
            "metrics": {
                "roc_auc_score": iso_roc,
                "average_precision": iso_avg_prec,
                **iso_threshold
            },
            "summary": {
                "alarm_rate": iso_test_alarm.mean(),
                "notes": "Yuksek skorlu örnekler anomaliye daha yakindir. Bunun icin - ile carpilmistir"
            }
        },
        "local_outlier_factor": {
            "parameters": {
                "n_neighbors": 30,
                "contamination": "auto",
                "novelty": True # yeni veri eklendikce de calisir
            },
            "metrics": {
                "roc_auc_score": lof_roc,
                "average_precision": lof_avg_prec,
                **lof_threshold
            },
            "summary": {
                "alarm_rate": lof_test_alarm.mean(),
                "notes": "Yuksek skorlu ornekler anomaliye daha yakindir."
            }
        }
    },
    # Eger ki olusturulma zamanini gormek isterseniz
    # "metadata_created_at": pd.Timestamp.now().isoformat(),
    "general_notes": [
        "Veri olceklemesi bu calismada kullanilmamistir.",
        "Threshold (esik deger) secimi F1 skoruna göre optimize edilmistir.",
        "ROC-AUC yerine Average Precision Score, dengesiz veri setlerinde daha anlamlidir."
    ]
}

    # kaydedelim
    with open(OUTLIER_META_CSV, mode = 'w', encoding ='utf-8') as file:
        json.dump(meta_data
                  ,file
                  ,indent= 4
                  ,ensure_ascii= False)
        print(f"Meta Degerlerin Bulundugu Klasor Yolu: {OUTLIER_META_CSV}")
    return iso_result_dict, lof_result_dict

#%% ana fonksiyonda cagiralim
if __name__ == '__main__':
    ensure_processed_dir()
    iso_result, lof_result = main()
    