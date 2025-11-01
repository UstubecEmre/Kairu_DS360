#%% Gerekli Kutuphanelerin Dahil Edilmesi (import required libraries)
import os
import sys
import argparse
from datetime import datetime
import numpy as np
import pandas as pd



""" Adim Adim Onceden Hazirlanmis Olan Pipeline'i Calistirir

Ornek Kullanim:
python run_modular.py [--module ModulunuzunIsmi]

"""


#%% Veri on hazirligi
def run_p1_data_preparation():
    """P1: Veri On Hazirlama Modulu"""
    print(f"P1: Data Preparation Baslatiliyor...")
    try:
        from P1_data_preparation.create_m5_subset import main as create_subset
        print(f"M5 Alt Kumesi Olusturuluyor...")
        create_subset()
        print("P1 Modulu Tamamlandi")
        return True
    except Exception as err:
        raise Exception(f"Veri On Hazirlama Modulunde Bir Sorun Olustu: {err}")
    
def run_p2_feature_engineering():
    """P2: Feature Engineering Modulu"""
    print("Feature Engineering (Oznitelik Muhendisligi) Modulu Baslatiliyor...")
    try:
        from P2_feature_engineering.feature_engineering import main as create_features
        print("Oznitelikler Olusturuluyor...")
        create_features()
        print("P2 Feature Engineering Modulu Tamamlandi")
        return True
    except Exception as err:
        raise Exception(f"P2 Feature Engineering Modulu Basarisiz Oldu: {err}")
    

def run_p3_traditional_models():
    """P3: Traditional Models Modulu"""
    print(f"P3: ARIMA Model Baslatiliyor...")
    try:
        from P3_traditional_models.arima_single_item import main as run_arima
        print("ARIMA Model Calistiriliyor...")
        run_arima()
        print("P3 Geleneksel Arima Modulu Tamamlandi")
        return True
    except Exception as err:
        raise Exception(f"P3: Traditional Models Modulu Basarisiz Oldu: {err}")
    
def run_p4_modern_models():
    """P4: Modern Models Modulu"""
    print(f"P4: Modern Models Modulu Baslatiliyor...")
    try:
        from P4_modern_models.prophet_single_item import main as run_prophet
        print("Prophet Modeli Calistiriliyor...")
        run_prophet()
        print("P4: Modern Models Modulu (Prophet) Tamamlandi")
        return True 
    except Exception as err:
        raise Exception(f"P4: Modern Models (Prophet) Modulu Basarisiz Oldu: {err}")
    
def run_p5_ml_models():
    """P5: ML Models Modulu"""
    print(f"P5: ML Models Modulu Baslatiliyor...")
    try:
        from P5_ml_models.lightgbm_multi_items import main as run_lightgbm
        print(f"P5: ML Models (LightGBM) Modulu Baslatiliyor...")
        run_lightgbm()
        return True
    except Exception as err:
        raise Exception(f"P5: ML Models (LightGBM) Modulu Basarisiz Oldu: {err}")
    
def main():
    """Ana Fonksiyon"""
    parser = argparse.ArgumentParser(description= 'M5 Tahminleyici Moduler Pipeline')
    parser.add_argument("--module"
                        ,type= str
                        ,choices= ['P1', 'P2', 'P3', 'P4', 'P5']
                        ,help= "Istenilen Modullu Kendi Basina Calistirir")
    args = parser.parse_args()
    
    print(f"M5 Tahminleyici - Moduler Pipeline'i")
    print(f"Baslangic Zamani: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python Versiyonu: {sys.version.split()[0]}")
    print(f"Mevcut Calisma Dizini: {os.getcwd()}")
    
    # Modul fonksiyonlarini atayalim
    module_functions = {
        "P1":run_p1_data_preparation
        ,"P2": run_p2_feature_engineering
        ,"P3": run_p3_traditional_models
        ,'P4': run_p4_modern_models
        ,'P5': run_p5_ml_models
    }
    # Basarili ve basarisiz 
    success_count = 0
    total_count = 0
    failed_modules = []
    
    if args.module:
        """Spesifik Olarak Istenilen Modulu Tek Basina Calistirir"""
        print(f"Sadece {args.module} Modulu Calistiriliyor...")
        func = module_functions[args.module]
        success = func()
        success_count = 1 if success else 0
        total_count = 1
    else:
        print("Full Pipeline Calistiriliyor")
    
        for module_name, func in module_functions.items():
            print(f"{module_name} Modulu Calistiriliyor")
            start_time = datetime.now()
            try:
                success = func()
                end_time = datetime.now()
                duration = end_time - start_time
                print(f"{module_name} Modulu {duration} Surede Tamamlandi")
                
                if success:
                    success_count += 1
                else:
                    failed_modules.append(module_name)
                    
            except Exception as err:
                end_time = datetime.now()
                duration = end_time - start_time
                print(f"{module_name} Modulu Hata Verdi. Calisma Suresi: {duration}: {err}")
                failed_modules.append(module_name)
            
            total_count += 1
            print("")
    
    print("*"*50)
    print(f"Pipeline Tamamlandi")
    print(f"Basarili Moduller: {success_count} / {total_count}")
    print(f"Bitis Zamani: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if failed_modules:
        print(f"Basarisiz Moduller: {', '.join(failed_modules)}")
    
    
    # Basarisiz modul var mi, kontrol edelim
    if success_count == total_count:
        print("Tum Moduller Basariyla Gerceklesti")
        return 0
    else:
        print("Bazi Modullerde Hata Olustu")
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)