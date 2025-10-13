#%% import required libraries (Gerekli kutuphane modullerini iceri aktar)
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, f1_score, accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import json 



#%% define a class (Bir sinif tanimlayalim)
class ModelEvaluater():
    
    # init method
    def __init__(self, model = None, model_name = "Model"):
        self.model = model
        self.model_name = model_name
        self.evaluation_scores = {}
        
    def calc_metrics(self, X_test, y_true, y_pred_proba = None, threshold = 0.5):
        if self.model is None and y_pred_proba is None:
            raise ValueError("Gecerli Bir Model Ismi ve Olasiliksal Tahmin Degeri Giriniz")
        
        if y_pred_proba is None:
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        
        # y_pred_proba = self.model.predict_proba(X_test)[:, 1] # positive'leri alsin
        # esik deger olarak 0.5 belirledik, bu deger artirilabilir
        y_pred = (y_pred_proba >=threshold).astype(int) 
        prec, rec, threshold = precision_recall_curve(y_true, y_pred_proba)
        evaluate_scores = {
            "accuracy": accuracy_score(y_true, y_pred)
            ,"precision": precision_score(y_true, y_pred)
            ,"recall": recall_score(y_true, y_pred)
            ,"f1": f1_score(y_true, y_pred)
            ,"confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
            ,"roc_auc": roc_auc_score(y_true, y_pred_proba)
            ,"precision_curve": prec.tolist()
            ,"recall_curve": rec.tolist()
            ,"threshold": threshold.tolist()
        }   
        self.evaluation_scores = evaluate_scores
        return evaluate_scores
    
    def plot_confusion_matrix(self, X_test, y_true, y_pred_proba = None, threshold = 0.5):
        if y_pred_proba is None and self.model is not None:
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        y_pred = (y_pred_proba >= threshold).astype(int)
        conf_mat = confusion_matrix(y_true, y_pred)
        disp_conf_mat = ConfusionMatrixDisplay(conf_mat)
        disp_conf_mat.plot(cmap = "coolwarm", fmt = "%d" )
        
    
    
    def print_evaluation(self, save_path = False):
        print(json.dumps(self.evaluation_scores, indent = 4, ensure_ascii= False))
        if save_path:
            with open(save_path, mode = 'w', encoding = 'utf-8') as file:
                json.dump(self.evaluation_scores, file, indent = 4, ensure_ascii= False)
            print(f"Degerlendirme Sonuclari Kaydedildi. Dosya Yolu: {save_path}")
        