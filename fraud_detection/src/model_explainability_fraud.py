#%% import required libraries (Gerekli modulleri iceri aktar)
# for data manipulation and visualization (Veri manipulasyonu ve gorsellestirme icin)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# for warnings (uyarilari gormezden gelmek icin)
import warnings
warnings.filterwarnings('ignore')


import logging
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)
# for modelling
from sklearn.ensemble import RandomForestClassifier

# feature importances
from sklearn.inspection import permutation_importance
from download_data import load_data

fraud_df = load_data()

#%% ensure shap and lime import
"""
Global tanimla
SHAP_AVAILABLE = False
LIME_AVAILABLE = False

# fonksiyonlar disarida erisilemiyor, local'de kaliyor, mantikli degil
def ensure_shap_imported():
    global SHAP_AVAILABLE
    try:
        import shap
        SHAP_AVAILABLE = True
        logger.info(f"shap Modulu Bulunuyor. Versiyon: {shap.__version__}")
    except ImportError:
        SHAP_AVAILABLE = False 
    except Exception as err:
        raise Exception(f"Beklenmeyen Bir Hata Olustu: {err}")

def ensure_lime_imported():
    global LIME_AVAILABLE
    try:
        import lime 
        from lime.lime_tabular import LimeTabularExplainer
        LIME_AVAILABLE = True 
        logger.info(f"Lime Modulu Bulunuyor. Versiyon: {lime.__version__}")
    except ImportError:
        LIME_AVAILABLE = False
    except Exception as err:
        raise Exception(f"Beklenmeyen Bir Hata Olustu: {err}")
""" 

try:
    import shap 
    SHAP_AVAILABLE = True
    logger.info(f"Shap Modulu Import Edildi. Versiyonu: {shap.__version__}")
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning(f"Shap Modulu Bulunamadi")
except Exception as err:
    logger.warning(f"Beklenmeyen Bir Hata Olustu: {err}")


try:
    import lime 
    LIME_AVAILABLE = True 
    logger.info(f"Lime Modulu Import Edildi.Kullanilan Versiyon: {lime.__version__}")
except ImportError:
    logger.warning("Lime Modulu Import Edilmemis")
    LIME_AVAILABLE = False
except Exception as err:
    logger.warning(f"Beklenmeyen Bir Hata Olustu: {err}")

class SimpleModelExplainer:
    
    # initial method / constructor
    def __init__(self, model, X_train, feature_names = None, class_names = None):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names or [f'feature_{col}' for col in range(X_train.shape[1])]
        self.class_names = class_names or ['Normal', 'Fraud']
        self.shap_explainer = None 
        self.shap_values = None
        # self.lime_explainer = None
    
    # define a method
    def initialize_shap(self, explainer_type = 'tree'):
        """ Agac tabanli model kullanarak SHAP dahil eder"""
        # ensure_shap_imported()
        if not SHAP_AVAILABLE:
            logger.warning("SHAP Modulu Import Edilmemis")
            return False
        
        try:
            if explainer_type == 'tree':
                self.shap_explainer = shap.TreeExplainer(self.model)
            else:
                background = shap.sample(self.X_train
                              ,min(100, len(self.X_train)))
                self.shap_explainer = shap.KernelExplainer(
                    self.model.predict_proba
                    ,background
                )
            logger.info("SHAP Explainer Kullanilabilir Halde")
            return True # programdan cikar
        except Exception as err:
            logger.error(f"Beklenmeyen Bir Hata Olustu: {err}")    
            
    
    def compute_shap_values(self, X_test, max_samples = 100):
        """SHAP degerini hesaplar"""
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            logger.warning("Kullanilabilir SHAP Modulu ve Aciklayicisi Bulunamadi")
            return None, X_test
        
        try:
            self.shap_values = self.shap_explainer.shap_values(X_test)
            if isinstance(self.shap_values, list):
                self.shap_values = self.shap_values[1] # Fraud class
            return self.shap_values, X_test
        except Exception as err:
            logger.error(f"Beklenmeyen Bir Hata Olustu: {err}")
            return None, X_test
    
    
    def plot_shap_summary(self, X_test = None):
        if not SHAP_AVAILABLE or self.shap_values is None:
            logger.warning("SHAP Modulu veya Aciklamasi Bulunmamaktadir")
            return False 
        
        plt.figure(figsize = (12, 8))
        shap.summary_plot(
            self.shap_values
            ,X_test
            ,feature_names = self.feature_names
            ,plot_type = 'bar'
            ,show = False
        )
        plt.title("SHAP ile Degiskenlerin Onemlilik Derecesi (Feature Importances With SHAP)")
        plt.tight_layout()
        plt.show()
        return True 
    
    def show_feature_importances(self, X_test, y_test):
        """ Permutaion Features kullanilarak degiskenlerin onem dereceleri belirlenir"""
        
        try:
            perm_importance = permutation_importance(
                estimator= self.model
                ,X = X_test
                ,y = y_test
                ,random_state= 42
                ,n_repeats = 5
                ,n_jobs= -1
            )
            # en onemli 10 ozelligi siralayip, gorsellestirelim
            sorted_indexes = np.argsort(perm_importance.importances_mean)[-10:]
            plt.figure(figsize = (12, 8))
            sns.barplot(
                y = [self.feature_names[i] for i in sorted_indexes]
                ,x = perm_importance.importances_mean[sorted_indexes]
                ,palette ="RdGn")
            plt.title("Feature Importance (Degiskenlerin Onem Derecesi)")
            plt.xlabel("Permutation Importance (Permutasyon Onem Derecesi)")
            plt.tight_layout()
            plt.show()
            
            return perm_importance
        except Exception as err:
            logger.error(f"Permutation Importance Error (Permutasyon Onem Hatasi): {err}")
            return None 
        
        
        
def demo_explainability():
    """ RandomForestClassifier modeli kullanilarak degiskenlerin onem derecesini belirler ve model aciklanabilirligi saglar """
    from sklearn.model_selection import train_test_split
    # from sklearn.datasets import make_classification
        
    logger.info("Model Aciklanabilirligi Basliyor...")
        
    X = fraud_df.drop('Class', axis = 1)
    y = fraud_df['Class']
        
    X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size= 0.2
            ,random_state = 42
            ,stratify = y
        )
    model = RandomForestClassifier(n_estimators = 100 ,random_state= 42)
    model.fit(X_train, y_train)
    print(f"Model Dogruluk Degeri: {model.score(X_test, y_test):.4f}")
        
        
        # Explainer olustur
    explainer = SimpleModelExplainer(
            model, X_train
        )
    
    if explainer.initialize_shap():
        shap_values, X_test = explainer.compute_shap_values(X_test, max_samples = 1000)
        if shap_values is not None:
            explainer.plot_shap_summary(X_test)
            
    # explainer.show_feature_importances(X_test, y_test)
    explainer.show_feature_importances(X, y) # tum veri seti icin
    print("Model Aciklanabilirligi Tamamlandi...")
    return explainer


#%% ana fonksiyonda cagir

if __name__ == '__main__':
    demo_explainability()