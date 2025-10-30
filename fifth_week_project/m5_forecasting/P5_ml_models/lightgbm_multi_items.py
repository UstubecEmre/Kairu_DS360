#%% Gerekli Kutuphanelerin Iceriye Dahil Edilmesi (import required libraries)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from datetime import timedelta, datetime

# model on isleme adimlari icin
from sklearn.preprocessing import LabelEncoder

# model metrikleri
from sklearn.metrics import mean_absolute_error, mean_squared_error

import warnings
warnings.filterwarnings('ignore')


try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    print("LightGBM Dahil Edilmemis. Lutfen Dahil Ediniz")



#%% 
class LightGBMModelMultiItemForecaster:
    """LightGBM Modeli ile Coklu Urun Satis Tahmini Gerceklestirir"""
    def __init__(self,  artifacts_path:str = r'D:\Kairu_DS360_Projects\fifth_week_project\m5_forecasting\artifacts'):
        self.artifacts_path = os.path.abspath(artifacts_path)
        self.train_df = None
        self.valid_df = None
        self.model = None
        self.label_encoders = {}
        self.feature_cols = []
        self.metrics = {}
        self.feature_importance = None
        
        # Klasorleri olusturalim
        os.makedirs(f'{artifacts_path}/models', exist_ok = True)
        os.makedirs(f'{artifacts_path}/predictions', exist_ok = True)
        os.makedirs(f'{artifacts_path}/figures', exist_ok = True)
        
    
    def load_feature_data(self):
         # Parquet dosyalarini ilgili klasorden yukleyelim 
        try: 
            train_path = f'{self.artifacts_path}/datasets/fe_train.parquet'
            valid_path = f'{self.artifacts_path}/datasets/fe_valid.parquet'

            if not os.path.exists(valid_path) or not os.path.exists(train_path):
                raise FileNotFoundError(f"Feature Dataset Dosya Yollari Bulunamadi")
            
            self.train_df = pd.read_parquet(train_path)
            
            self.valid_df = pd.read_parquet(valid_path)
            
            print(f"Egitim Seti Boyutu: {self.train_df.shape}")
            print(f"Dogrulama Seti Boyutu: {self.valid_df.shape}")
            
            print(f"Oznitelikleri: {list(self.train_df.columns)}")
            
            
            train_items = self.train_df['item_id'].unique()
            valid_items = self.valid_df['item_id'].unique()
            
            print(f"Egitim Seti Urun Sayisi: {train_items}")
            print(f"Dogrulama Seti Urun Sayisi: {valid_items}")
        except FileNotFoundError:    
            print("Ilgili Dosya Yollari Bulunamadi")
        
        except Exception as err:
            print(f"Beklenmeyen Bir Hata Olustu: {err}")
            
    def encode_categorical_features(self):
        """Kategorik Verileri Sayisal Hale Donusturur"""
        print("Kategorik Ozellikler Sayisal Hale Donusturuluyor...")
        categorical_cols = ['store_id', 'item_id']
        for col in categorical_cols:
            if col in self.train_df.columns:
                print(f"{col} Sutunu Encode Ediliyor...")
                
                # Encoder => LabelEncoder
                le = LabelEncoder()
                
                
                # Egitim verisini fit_transform edelim, dogrulama da sadece transform
                self.train_df[f'{col}_encoded'] = le.fit_transform(self.train_df[col])
                
                try:
                    self.valid_df[f'{col}_encoded'] = le.transform(self.valid_df[col])
                
                except ValueError as err:
                    print(f"{col} Sutunu Icin Gorulemeyen Degerler Var. 0 Atamasi Gerceklestirilecektir")   
                    valid_encoded = []
                    
                    for val in self.valid_df[col]:
                        
                        if val in le.classes_:
                            valid_encoded.append(le.transform([val])[0])

                        else:
                            valid_encoded.append(0)
                    self.valid_df[f'{col}_encoded'] = valid_encoded
                
                # Encoder nesnemizi saklayalim
                self.label_encoders[col] = le
                print(f"Egitim Verisi Benzersiz Degerleri: {self.train_df[f'{col}_encoded'].nunique()}")
                print(f"Dogrulama Verisi Benzersiz Degerleri: {self.valid_df[f'{col}_encoded'].nunique()}")
            
            print(f"Encode Edilen Kategorik Degisken Sayisi: {len(categorical_cols)}")
            
    def prepare_features_target(self):
        """LightGBM Modeli Icin Oznitelikleri ve Hedef Degiskeni Hazirlar"""
        print(f"LightGBM Modeli Icin Oznitelikler ve Hedef Degisken Hazirlaniyor...")
        
        # Hedef Degisken
        target_col = 'sales'
        
        # Lag Sutunlari
        lag_cols = [col for col in self.train_df.columns if 'lag_' in col]
        
        # Rolling Sutunlari
        roll_cols = [col for col in self.train_df.columns if 'rolling_' in col]
        
        # Tarih Sutunlari
        date_cols = ['dow', 'dom', 'weekofyear', 'month']
        
        
        # Encoded cols
        encoded_cols = [col for col in self.train_df.columns if '_encoded' in col]
        
        self.feature_cols = lag_cols + roll_cols + date_cols
        
        print("Ozniteliklerin Ozellikleri Gosteriliyor...")
        print(f"Lag Ozellikleri: {lag_cols}; Ozellik Sayisi: {len(lag_cols)}")
        print(f"Rolling Ozellikleri: {roll_cols}; Ozellik Sayisi: {len(roll_cols)}")
        print(f"Tarih (Date) Ozellikleri: {date_cols}; Ozellik Sayisi: {len(date_cols)}")
        print(f"Kategorik Ozellikleri: {encoded_cols}; Ozellik Sayisi: {len(encoded_cols)}")
        print(f"Toplam Ozellik Sayisi: {len(self.feature_cols)}")
        
        
        # Egitim ve Dogrulama Ayrimini Gerceklestirelim
        X_train = self.train_df[self.feature_cols].copy()
        y_train = self.train_df[target_col].copy()
        
        X_valid = self.valid_df[self.feature_cols].copy()
        y_valid = self.valid_df[target_col].copy()
        
        print(f"Egitim Verisi Boyutu: {X_train.shape}")
        print(f"Egitim Verisi Hedef Degisken Boyutu: {y_train.shape}")
        
        print(f"Dogrulama Verisi Boyutu: {X_valid.shape}")
        print(f"Dogrulama Verisi Hedef Degiskenin Boyutu: {y_valid.shape}")


        # toplam eksik deger sayisi
        train_nans = X_train.isnull().sum().sum()
        valid_nans = X_valid.isnull().sum().sum()
        
        if train_nans > 0 or valid_nans > 0:
            print(f"Eksik Degerler Bulunmaktadir. Egitim Seti Icinde: {train_nans} ; Dogrulama Seti Icinde: {valid_nans}")
            print("Eksik Degerler 0 ile Dolduruluyor")
            X_train = X_train.fillna(0)
            X_valid = X_valid.fillna(0)
            print("Doldurma Islemi Tamamlandi.")
        return X_train, y_train, X_valid, y_valid
    
    def train_lightgbm_model(self, X_train, y_train, X_valid, y_valid):
        """LightGBM Modelini Egitir"""
        print("LightGBM Model Egitimi Baslatiliyor...")
        try:
            lgb_params = {
                'objective': 'regressive'
                ,'metric': 'rmse'
                ,'boosting_type': 'gbdt'
                ,'num_leaves': 31 
                ,'learning_rate': 0.05
                ,'feature_fraction': 0.9
                ,'bagging_fraction': 0.8
                ,'bagging_freq': 5
                ,'verbose': -1
                ,'random_state': 42
            }
            print("Model Parametreleri Yazdiliyor...")
            for key, value in lgb_params.items():
                print(f"{key} Degeri : {value}")
                
            # Veri setlerini olusturalim
            train_data = lgb.DataSet(X_train, label = y_train)
            valid_data = lgb.DataSet(X_valid, label = y_valid, reference = train_data)
            
            print("LightGBM Model Egitimi Baslatiliyor")
            
            self.model = lgb.train(
                lgb_params
                ,train_data
                ,valid_data
                ,valid_names = ['train', 'valid']
                ,num_boost_round = 500
                ,callbacks = [
                    lgb.early_stopping(stopping_rounds = 50)
                    ,lgb.log_evaluation(period = 100)] # erken durdurma  
            )
            print("LightGBM Model Egitimi Tamamlandi")
            print(f"En Iyi Iterasyon: {self.model.best_iteration}")
            print(f"Egitim Seti RMSE Degeri: {self.model.best_score['train']['rmse']}")
            print(f"Dogrulama Seti RMSE Degeri: {self.model.best_score['valid']['rmse']}")
            
        except Exception as err:
            print(f"Model Egitiminde Beklenmeyen Hata Gerceklesti: {err}")
        
        
    def calculate_validation_metrics(self, X_valid, y_valid):
        """Dogrulama Seti Metrik Degerlerini Hesaplar"""
        print(f"Dogrulama Seti Degerlendirme Metrikleri Hesaplaniyor...")
        
        y_pred = self.model.predict(X_valid, num_iteration = self.model.best_iteration)
        
        # Negatif degerleri 0 ile degistir
        y_pred = np.maximum(y_pred, 0)
        
        # MAE, MSE hesapla
        mae = mean_absolute_error(y_valid, y_pred)
        rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
        
        # MAPE
        mask = y_valid != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_valid[mask] - y_pred[mask]) / y_valid[mask])) * 100
        else:
            # Hesaplanamazsa sonsuz degeri ata
            mape = float('inf')
        
        # sMAPE
        denominator = (np.abs(y_valid) + np.abs(y_pred)) / 2
        mask = denominator != 0
        
        if mask.sum() > 0:
            smape = np.mean(np.abs(y_valid[mask] - y_pred[mask]) / denominator[mask]) * 100
        else:
            smape = float('inf')
        
        
        self.metrics = {
            "MAE": mae
            ,'RMSE': rmse
            ,'MAPE': mape
            ,'sMAPE': smape
            ,'R2': 1 - (np.sum((y_valid - y_pred) ** 2) / np.sum((y_valid - np.mean(y_valid)) ** 2))
        }
        
        
        print("Degerlendirme Metrikleri Hesaplandi. Metrik Degerleri Gosteriliyor...")
        print(f"Ortalama Mutlak Hata (MAE): {mae:.4f}")
        print(f"Kok Ortalama Kare Hata (RMSE): {rmse:.4f}")
        print(f"Ortalama Mutlak Yuzde Hata (MAPE): {mape:.4f}")
        print(f"Simetrik Ortalama Mutlak Yuzde Hata (sMAPE): {smape:.4f}")
        print(f"R²:{self.metrics['R2']:.4f}")

        return y_pred
    
    def create_feature_importance_plot(self):
        """Ozniteliklerin Onem Derecesini Gorsellestirir"""
        print(f"Ozniteliklerin Onem Dereceleri Gorsellestiriliyor...")
        
        # gain 
        importance_gain = self.model.feature_importance(importance_type = 'gain')
        feature_names = self.feature_cols
        
        # df olusturalim
        importance_df = pd.DataFrame(
            {
                'feature': feature_names
                ,'importance': importance_gain
            }
        ).sort_values(
            'importance'
            ,ascending= False)

        self.feature_importance = importance_df
        
        # Grafik olusturalim
        plt.figure(figsize = (10, 6))
        
        # En Etkili 15 Ozellik
        top_features = importance_df.head(15)
        
        bars = plt.barh(top_features['feature']
                        ,top_features['importance']
                        ,color = 'skyblue'
                        ,alpha = 0.7
        )
        
        # Degerleri sutun grafige yazdiralim
        for bar, value in zip(bars, top_features['importance']):
            plt.text(bar.get_width() + max(top_features['importance']) * 0.01
                    ,bar.get_y() + bar.get_height() / 2
                    ,f'{value:.2f}'
                    ,ha = 'left'
                    ,va = 'center'
                    ,fontweight = 'bold'
            )
        plt.title("LightGBM Modeli Onem Dereceleri", fontsize = 16, fontweight = 'bold')
        plt.xlabel('Onem Derecesi', fontsize = 14, fontweight = 'bold')
        plt.ylabel('Oznitelikler', fontsize = 14, fontweight = 'bold')
        plt.grid(True, alpha = 0.4)
        plt.tight_layout()

        # Kaydedelim
       
        importance_path = f'{self.artifacts_path}/figures/lgbm_feature_importance.png'
            
        plt.savefig(importance_path, dpi=300, bbox_inches='tight')
        print(f"Oznitelikerin Onem Dereceleri Sutun Grafigi Kaydedildi: {importance_path}")
        plt.close()
        
        
        print("En Onemli 10 Ozellik Gosteriliyor...")
        for idx, (_, row) in enumerate(top_features.head(10).iterrows(), 1):
            print(f"{idx:2d}. {row['feature']:<20}: {row['importance']:6.0f}")
                
    
    def iterative_forecast(self, forecast_steps = 28):
        """Iterative Olarak 28 Gunluk Satis Tahmini Gerceklestirir"""
        print(f"{forecast_steps} Gunluk Iterative Satis Tahmini Yapiliyor...")
        
        try:
            # Son ve ilk tarihleri al
            last_date = self.valid_df.index.max()
            forecast_start = last_date + timedelta(days = 1) 
            
            print(f"Son Veri Tarihi: {last_date}")
            print(f"Tahmin Baslangic Tarihi: {forecast_start}")
            
            # Tum Urunler Icin Tahmin
            all_forecasts = []
            unique_items = self.valid_df['item_id'].unique()
            
            for item_id in unique_items:
                print(f"{item_id} ID'li Urun Icin Tahmin Gerceklestiriliyor")
                
                # Dogrulama verisinin son satirini alalim
                item_valid = self.valid_df[self.valid_df['item_id'] == item_id].copy()
                
                if len(item_valid) == 0:
                    print(f"{item_id} ID'li Urun Bulunamadi")
                    continue
                
                last_row = item_valid.iloc[-1].copy()
                
                item_forecasts = []
                current_features = last_row.copy()
                
                
                for step in range(forecast_steps):
                    forecast_date = forecast_start + timedelta(days = step)
                    # Tarih ozelliklerini guncelleyelim
                    # day of week
                    current_features['dow'] = forecast_date.weekday()
                    
                    # day of month
                    current_features['dom'] = forecast_date.day
                    
                    # week of year
                    current_features['weekofyear'] = forecast_date.isocalendar()[1]
                    
                    # month
                    current_features['month'] = forecast_date.month
                    
                    # Tahmin gerceklestirelim
                    X_pred = current_features[self.feature_cols].values.reshape(1, -1)
                    y_pred = self.model.predict(X_pred, num_iteration=self.model.best_iteration)[0]
                    y_pred = max(0, y_pred)  # Negatif degerleri 0 yapalim
                    
                    
                    # Sonucu kaydedelim
                    item_forecasts.append(
                        {
                            'date': forecast_date
                            ,'item_id': item_id
                            ,'store_id': last_row['store_id']
                            ,'y_pred': y_pred 
                        }
                    )
                    # Lag ozniteliklerini guncelleyelim
                    if 'lag_1' in current_features:
                        # degerleri 28 gun kaydiralim
                        if 'lag_28' in current_features:
                            pass
                
                all_forecasts.extend(item_forecasts)
            
            # df olusturalim
            forecast_df = pd.DataFrame(all_forecasts)
            
            print(f"Uretilen Tahmin Sayisi: {len(forecast_df)}")
            print(f"Benzersiz Urun Sayisi: {forecast_df['item_id'].nunique()}")
            print(f"Tahmin Degerleri Tarih Araligi: {forecast_df['date'].min()} - {forecast_df['date'].max()}")
            print(f"Ortalama Tahmin: {forecast_df['y_pred'].mean():.4f}")

            return forecast_df
        
        except Exception as err:
            raise Exception(f"Iterative Tahminleme Gerceklestirilemedi: {err}")
    
    def save_results(self, forecast_df):
        """Sonuclari Kaydeder"""
        print(f"LightGBM Modeli Icin Sonuclar Kaydediliyor")
        try:
            model_path = f'{self.artifacts_path}/models/lightgbm.pkl'
            model_data = {
                'model': self.model
                ,'feature_cols': self.feature_cols
                ,'label_encoders': self.label_encoders
                ,'metrics': self.metrics
                , 'feature_importance': self.feature_importance
                ,'model_params': {
                    'best_iteration': self.model.best_iteration
                    ,'best_score': self.model.best_score
                }
            }
            
            with open(model_path, 'wb') as file:
                pickle.dump(model_data, file)
            print(f"LightGBM Model Kaydedildi: {model_path}")
            
            pred_path = f'{self.artifacts_path}/predictions/lightgbm_forecast_all.csv'
            forecast_df.to_csv(pred_path, index = False)
            print(f"Tahmin Degerleri Kaydedildi: {pred_path}")
        
            # Ozet rapor olustur
            import json
            report = {
                'model_type': 'LightGBM'
                ,'training_date': datetime.now().isoformat()
                ,'data_info': {
                    'train_shape': list(self.train_df.shape)
                    ,'valid_shape': list(self.valid_df.shape)
                    ,'n_items': self.train_df['item_id'].nunique()
                    ,'feature_count': len(self.feature_cols)
                },
                'model_performance': self.metrics
                ,'top_features': self.feature_importance.head(10).to_dict('records') if self.feature_importance is not None else []
                ,'forecast_info': {
                    'forecast_steps': 28
                    ,'forecast_items': int(forecast_df['item_id'].nunique())
                    ,'total_predictions': len(forecast_df)
                }
            }
            report_path = f'{self.artifacts_path}/predictions/lightgbm_report.json'
            with open(report_path, 'w') as file:
                json.dump(
                    report
                    ,file
                    ,indent = 4
                    ,default= str 
                )
            print(f"Rapor {report_path} Dosya Yoluna Kaydedildi")
        except Exception as err:
            raise Exception(f"Model Kaydedilirken Hata Olustu: {err}")
    
    
    
    def run_lightgbm_pipeline(self):
        """LightGBM Modeli Icin Bastan Sona Bir Pipeline Hazirlar"""
        try:
            # Verinin yuklenmesi
            self.load_feature_data()
            
            # Verinin hazir hale getirilmesi
            
            # encode veya scale islemleri
            self.encode_categorical_features()
            
            X_train, y_train, X_valid, y_valid = self.prepare_features_target()
            
            # Modelin egitilmesi
            self.train_lightgbm_model(X_train, y_train, X_valid, y_valid)
                        
            # Modelin Tahminlemesi
            y_pred_valid = self.calculate_validation_metrics(X_valid, y_valid)
            
            self.create_feature_importance_plot()
            
            # Modelin degerlendirilmesi icin tahminleme yapilmasi
            forecast_df = self.iterative_forecast()
            
            # Sonuclarin Degerlendirilmesi ve kaydedilmesi
            self.save_results(forecast_df)

            print(f"Coklu Urun Tahminlemesi LightGBM Modeli Kullanilarak Gerceklestirildi")
            print(f"Hizli Bir Bakis Icin Ozet Bilgiler Getiriliyor...")
            print(f"Kullanilan Model Ismi: LightGBM Regressor")
            print(f"Dogrulama Seti Simetrik Mutlak Ortalama Yuzdesel Hata: {self.metrics['sMAPE']:.4f}")
            print(f"R² Skoru: {self.metrics['R2']:.4f}")
            print(f"Gerceklestirilen Tahmin Miktari: {len(forecast_df)} Adet\n28 Gun * {forecast_df['item_id'].nunique()} Urun")
            print(f"Ciktilarin Bulundugu Dosya Yolu: {self.artifacts_path}/")
            
            return self.model, forecast_df, self.metrics
        
        except Exception as err:
            raise Exception(f"LightGBM Modeli Pipeline'i Basarisiz Oldu: {err}")

def main():
    """LightGBM Modeli ile Tahminlemeyi Gerceklestiren Ana Fonksiyondur"""
    if not LIGHTGBM_AVAILABLE:
        print("LightGBM Modeli Icin Ilgili Kutuphaneyi Dahil Ediniz")
        return
    
    try:
        forecaster = LightGBMModelMultiItemForecaster()
        
        # Pipeline'i calistiralim
        model, forecast, metrics = forecaster.run_lightgbm_pipeline()
        print("LightGBM ile Coklu Urun Tahminlemesi Basariyla Gerceklestirildi")
    except KeyboardInterrupt:
        print("Kullanici Tarafindan Uygulama Durduruldu")
    except Exception as err:
        print(f"LightGBM Pipeline Sureci Basarisizlikla Sonuclandi: {err}")

if __name__ == '__main__':
    main()