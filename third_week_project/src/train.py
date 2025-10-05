#%% import required libraries (Gerekli kutuphaneleri import et)
import os # for operating system related operations (isletim sistemi ile ilgili islemler icin)

import pandas as pd # for data manipulation (veri manipule etme icin)
import numpy as np #for numerical operations (sayisal islemler icin)
import matplotlib.pyplot as plt # for data visualization (veri gorsellestirme icin)
import seaborn as sns # for statistical data visualization (istatistiksel veri gorsellest

from preprocessing import get_split_data 

# for model building (model olusturma icin)
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# for resampling (yeniden ornekleme icin)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
# from preprocessing import preprocess_loan_data

# for ignoring warnings (uyarilari yoksayma icin)
from warnings import filterwarnings
filterwarnings('ignore')

# for saving models and encoders (model ve encoder kaydetme icin)
import joblib
import json

from sklearn.pipeline import Pipeline
from preprocessing import ensure_processed_dir, get_split_data, calculate_class_ratio

""" 
Amac: Loan veri seti uzerinde model egitimi yapmak.
Dengesiz veri seti ile bas etmek icin SMOTE ve RandomUnderSampler kullan.
Burada class_weight = 'balanced' parametresinin nasil kullanilabilecegini goster.
XGBoost modelinde scale_pos_weight parametresinin nasil kullanilabilecegini goster.
Sentetik veri olusturmak icin SMOTE kullan.
Ilgili dosyalari kaydet: artifacts isminde bir klasor olustur ve icerisine model, encoder ve
ozellik isimlerini kaydet.
"""
# ARTIFACTS_DIR = 'artifacts'

#%% create artifacts directory if not exists (artifacts klasoru yoksa olustur)
def ensure_artifacts_dir():
    artifacts_dir = 'artifacts'
    os.makedirs(artifacts_dir, exist_ok=True)
    return artifacts_dir








#%% evaluate model function (model degerlendirme fonksiyonunu tanimla)
def evaluate_and_print_model(title:str, y_test:np.ndarray, y_pred:np.ndarray, y_proba:np.ndarray):
    if title is None or not isinstance(title, str):
        print("Model basligi gecerli bir string olmalidir.")
        return False
    if y_test is None or y_pred is None or y_proba is None:
        print("y_test, y_pred ve y_proba gecerli degerler olmalidir.")
        return False
    print("="*80)
    print(f"***** {title} *****")
    print("Siniflandirma Raporu (Classification Report):\n",classification_report(y_test, y_pred, digits= 4))
    
    report = classification_report(y_test, y_pred, digits=4, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Karmasiklik Matrisi Gosteriliyor...\n{conf_matrix}") 
    try:
        if y_proba.ndim > 1 and y_proba.shape[1] > 1:
            y_proba = y_proba[:, 1] # positive class probabilities (pozitif sinifin olasiligi)
        print("Egri Altindaki Alan (AUC):", roc_auc_score(y_test, y_proba).round(4))
        auc = roc_auc_score(y_test, y_proba).round(4)
    except Exception as err:
        auc = None
        print(f"AUC Hesaplanamadi: {err}")
    print("="*80)
    # for returning values (degerleri dondurmek icin)
    return {
        "classification_report":report,
        "confusion_matrix":conf_matrix,
        "roc_auc":auc
    }


#%% train with different models and resampling techniques (farkli modeller ve yeniden ornekleme teknikleri ile egit)

def scenario_no_resampling(save_tag = '_cw'):
    """Herhangi bir yeniden ornekleme yapmadan model egitimi yapar.
    Kullanilacak makine ogrenme algoritmalari: Logistic Regression, Random Forest, XGBoost"""
    
    artifacts_dir = ensure_artifacts_dir()
    X_train, X_test, y_train, y_test, pre = get_split_data()
    
    # save feature names (ozellik isimlerini kaydet)
    with open(f'{artifacts_dir}/feature_schema.json',"w") as file:
        json.dump({"columns": list(X_train.columns)}, file, indent = 4)
    
    
    # 1- Apply preprocessor =>  Logistic Regression with class_weight = 'balanced'
    log_reg = Pipeline(steps = [
        ("pre",pre),
        ('clf',LogisticRegression(class_weight = 'balanced'
                                  ,max_iter = 15000))
    ]
        
    )
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    y_proba = log_reg.predict_proba(X_test)[:, 1] # positive class probabilities (pozitif sinifin olasiligi)
    
    # evaluate and print model results (model sonuclarini degerlendir ve yazdir)
    log_reg_results = evaluate_and_print_model("Logistic Regression with class_weight = 'balanced'"
                                              ,y_test
                                              ,y_pred
                                              ,y_proba)
    # save model and preprocessor (modeli ve on isleyiciyi kaydet)
    
    joblib.dump(log_reg,
                f"{artifacts_dir}/model_log_reg{save_tag}.pkl")
    
    joblib.dump(pre,
                f"{artifacts_dir}/preprocessor_log_reg{save_tag}.pkl")    
    

# 2 - Apply preprocessor => Random Forest with class_weight = 'balanced'
    # create Random Forest model with class_weight = 'balanced'
    rf_clf = Pipeline(steps = [
        ("pre",pre),
        ('clf',RandomForestClassifier(
            class_weight = "balanced"
            ,n_estimators = 150
            ,random_state = 42
            ,max_depth = 10
            ,min_samples_split = 5))
    ])

    #fit model (modeli egit)
    rf_clf.fit(X_train, y_train)
    y_pred = rf_clf.predict(X_test)
    y_proba = rf_clf.predict_proba(X_test)[:, 1] # positive class probabilities (pozitif sinifin olasiligi)
    # evaluate and print model results (model sonuclarini degerlendir ve yazdir)
    rf_clf_results = evaluate_and_print_model("Random Forest with class_weight = 'balanced'"
                                            ,y_test
                                            ,y_pred
                                            ,y_proba)
    # save model and preprocessor (modeli ve on isleyiciyi kaydet)
    
    joblib.dump(rf_clf,
                f"{artifacts_dir}/model_rf{save_tag}.pkl")
    
    joblib.dump(pre,
                f"{artifacts_dir}/preprocessor_rf{save_tag}.pkl") 
    
    
    # 3 - Apply preprocessor => XGBoost with scale_pos_weight
    _, _, pos_weight = calculate_class_ratio(y_train) # positive, negative, positive_weight
    # create XGBoost model with scale_pos_weight
    xgb = Pipeline(steps = [
        ("pre", pre),
        ("clf", XGBClassifier(
            n_estimators = 500
            ,random_state = 42
            ,max_depth = 7
            ,learning_rate = 0.05
            ,subsample = 0.8
            ,colsample_bytree = 0.8
            ,reg_lambda = 1,
            scale_pos_weight = pos_weight
            ,eval_metric = 'logloss'
        ))
    ])
    
    # fit model (modeli egit)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    y_proba = xgb.predict_proba(X_test)[:, 1] # positive class probabilities (pozitif sinifin olasiligi)
    xgb_results = evaluate_and_print_model("XGBoost with scale_pos_weight"
                                            ,y_test
                                            ,y_pred
                                            ,y_proba)
    
    # save model and preprocessor (modeli ve on isleyiciyi kaydet)
    joblib.dump(xgb,
                f"{artifacts_dir}/model_xgb{save_tag}.pkl")
    joblib.dump(pre,
                f"{artifacts_dir}/preprocessor_xgb{save_tag}.pkl")
    
    return {
        "log_reg": log_reg_results,
        "random_forest": rf_clf_results,
        "xgboost": xgb_results
    }


#%% define scenario_with_sampler()
# Veri dengesizligini onlemek icin Resampling gerceklestirelim.
# Onemli Not: (sadece TRAIN’e yani egitim setine uygulanmalidiri)
# Bu senaryonun kolon şeması da ayrıca kaydedilir (etiket: under veya smote).


def scenario_with_sampler(sampler, tag, return_results: bool = True):
    ensure_processed_dir()
    
    artifacts_dir = ensure_artifacts_dir()
    
    X_train, X_test, y_train, y_test, pre = get_split_data()
    
    
    with open(f"{artifacts_dir}/feature_schema_{tag}.json", mode= "w") as file:
        json.dump(
        {"columns": list(X_train.columns)}
        ,fp = file
        ,indent = 4)
    
    # Not: fit_transform sadece X_train'de uygulanir.
    # X_test'de sadece fit() uygulanir. Bu veri sizintisini onlemektedir.
    X_train_transformed = pre.fit_transform(X_train)
    X_transformed_resampled, y_resampled = sampler.fit_resample(X_train_transformed, y_train)
  
    
    # 1-build a Logistic Regression model (Lojistik regresyon modelini olustur)
    log_reg = LogisticRegression(
        max_iter = 1500
        ,C = 0.8
        ,tol = 0.001
    )
    
    # fit model(modeli egit)
    log_reg.fit(X_transformed_resampled, y_resampled)
    
    # apply only transform() function for test dataset (test veri seti icin transform uygula )
    X_test_transformed = pre.transform(X_test)
    y_pred_lr = log_reg.predict(X_test_transformed)
    y_proba_lr = log_reg.predict_proba(X_test_transformed)[:, 1]
    log_reg_result = evaluate_and_print_model(
        f"Logistic Regression model with ({sampler.__class__.__name__})"
        ,y_test
        ,y_pred_lr
        ,y_proba_lr)
    
    
    # build random forest classifier model
    rf = RandomForestClassifier(
        n_estimators = 150
        ,random_state = 42
        ,max_depth = 10
        ,min_samples_split = 5
    )
    
    # fit model
    rf.fit(X_transformed_resampled, y_resampled)
    y_pred_rf = rf.predict(X_test_transformed)
    y_proba_rf = rf.predict_proba(X_test_transformed)[:, 1] # positive
    rf_result = evaluate_and_print_model(
       f"Random Forest Classifier Model with ({sampler.__class__.__name__})"
       ,y_test
       ,y_pred_rf
       ,y_proba_rf
    )
    
    # build XGBoost model
    xgb = XGBClassifier(
        n_estimators = 500
        ,random_state = 42
        ,max_depth = 7
        ,learning_rate = 0.05
        ,subsample = 0.8
        ,colsample_bytree = 0.8
        ,reg_lambda = 1
        ,eval_metric = "logloss"
    )
    
    # fit model
    xgb.fit(X_transformed_resampled, y_resampled)
    y_pred_xgb = xgb.predict(X_test_transformed)
    y_proba_xgb = xgb.predict_proba(X_test_transformed)[:, 1] 
    xgb_result = evaluate_and_print_model(
        f"XGBoost Model with ({sampler.__class__.__name__})"
        ,y_test
        ,y_pred_xgb
        ,y_proba_xgb
    )
    
    # save all models
    joblib.dump(pre, 
                f"{artifacts_dir}/preprocessor_{tag}.pkl")
    
    joblib.dump(log_reg,
                f"{artifacts_dir}/model_log_reg_{tag}.pkl")
    
    joblib.dump(rf,
                f"{artifacts_dir}/model_rf_{tag}.pkl")
    
    joblib.dump(xgb,
                f"{artifacts_dir}/model_xgb_{tag}.pkl")
    
    if return_results:
        return {
            "scenario": tag or "no_resampling"
            ,"models":{
                "log_reg":log_reg_result
                ,"rf":rf_result
                ,"xgb":xgb_result
        }
        }

def main():
    """main fonksiyonunda bu fonksiyonlari cagiralim. Herhangi bir ornekleme olmayan versiyonu cagir
    Diger dengesiz veri ile basa cikma yontemlerinden SMOTE (sentetik veri uretimi) ile RandomUnderSampler yontemini gerceklestirelim.
    """
    # normal class weight
    print("Yeniden Ornekleme Olmadan Egitim Gerceklestiriliyor...")
    result_cw =scenario_no_resampling(save_tag = "_cw")
    
    # random undersampling    
    print("RandomUnderSampler Yontemi ile Egitim Gerceklestiriliyor...")
    result_undersampler = scenario_with_sampler(
        sampler = RandomUnderSampler(random_state=42)
        ,tag = "under"
    )
    
    # SMOTE
    print("Sentetik Veri Uretimi Yontemi ile Egitim Gerceklestiriliyor...")
    result_smote = scenario_with_sampler(
        sampler = SMOTE(random_state = 42)
        ,tag = 'smote'
    )
    

def save_metrics(metrics_dictionary, file_path = 'artifacts/metrics.json'):
    """Model performansini daha iyi degerlendirebilmek amaciyla metrik degerlerini kaydeder """
    os.makedirs(os.path.dirname(file_path), exist_ok= True)
    with open(file_path, mode = "w") as file:
        json.dump(metrics_dictionary, file, indent = 4)
    print(f"Metrikler kaydediliyor. Dosya yolu: {file_path}")
    
    
    
if __name__ == '__main__':
    # main()
    result_cw =scenario_no_resampling(save_tag = "_cw")
    result_undersampler = scenario_with_sampler(
        sampler = RandomUnderSampler(random_state=42)
        ,tag = "under"
    )
    result_smote = scenario_with_sampler(
        sampler = SMOTE(random_state = 42)
        ,tag = 'smote'
    )
    
    all_metrics = {
        "no_resampling":result_cw
        ,"undersampling":result_undersampler
        ,"smote": result_smote
    }
    save_metrics(all_metrics)