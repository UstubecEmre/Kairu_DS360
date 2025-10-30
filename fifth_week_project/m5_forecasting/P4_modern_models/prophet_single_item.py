""" 
Prophet, Meta (eski bilinen ismiyle Facebook) tarafindan gelistirilmis, modern, kullanici dostu bir yontemdir.
ARIMA modele gore mevsimsellikleri, tatil gunlerini de hesaba katarak modelleme gerceklestirmemize olanak saglamaktadir 
"""


#%% Gerekli Kutuphanelerin Dahil Edilmesi (import required libraries)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import json
from datetime import timedelta, datetime

from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

#%% Prophet dahil edelim

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    print(f"Prophet Kutuphanesi Bulunamadi. Lutfen Dahil Ediniz")
    PROPHET_AVAILABLE = False


#%% 
class ProphetModelSingleItemForecaster():
    def __init__(self, item_id = None, artifacts_path:str = r'D:\Kairu_DS360_Projects\fifth_week_project\m5_forecasting\artifacts'):
        self.artifacts_path = os.path.abspath(artifacts_path)
        self.item_id = item_id
        self.train_series = None
        self.valid_series = None
        self.prophet_model = None
        self.forecast_df = None
        self.metrics = {}
        self.arima_metrics = {}
        
        # Kaydedilmeleri icin klasor olusturalim
        
        os.makedirs(f'{artifacts_path}/models', exist_ok = True)
        os.makedirs(f'{artifacts_path}/predictions', exist_ok = True)
        os.makedirs(f'{artifacts_path}/figures', exist_ok = True)
    
    
    def load_arima_results(self):
        """Prophet ve ARIMA Modelleri Karsilastirilmasi Icin Gerekli Sonuclari Yukler"""
        print("Karsilastirma Yapabilmek Amaciyla ARIMA Yontemi Sonuclari Yukleniyor...")
        try:
            arima_files = [file for file in os.listdir(f'{self.artifacts_path}/models') if file.startswith('arima_')]
            
            # dosya var mi?
            if not arima_files:
                print("Maalesef ARIMA Modeli Bulunamadi. Lutfen 'arima_single_item.py' Python Dosyasini Calistirin")
                return None 
            
            arima_file = arima_files[0]
            self.item_id = arima_file.replace('arima_', '').replace('.pkl', '')
            print(f"ARIMA Modelinden 'item_id' Bilgisi Aliniyor. 'item_id': {self.item_id}")
            
            
            print("ARIMA Raporunuz Getiriliyor...")
            arima_report_path = f'{self.artifacts_path}/predictions/arima_report_{self.item_id}.json'
            # okuma modunda acalim
            with open(arima_report_path, 'r') as file:
                arima_report = json.load(file)
            
            
            # Metriklere dict uzerinden ulasalim
            self.arima_metrics = arima_report['metrics']
            print("ARIMA Modeli Degerlendirme Metrikleri Getiriliyor...")
            print(f"Ortalama Mutlak Hata Degeri (MAE): {self.arima_metrics['MAE']:.4f}")
            print(f"Simetrik Ortalama Mutlak Yuzdesel Hata Degeri (sMAPE): {self.arima_metrics['sMAPE']:.4f}")
        
            return arima_report
        
        except Exception as err:
            raise Exception(f"Karsilastirilacak ARIMA Modeli Getirilirken Beklenmeyen Bir Hata Olustu: {err}")


    def load_time_series(self):
        """Zaman Serisi Yukler"""
        print(f"{self.item_id} Urunu Icin Zaman Serisi Uretiliyor...")
        try:
            # egitim ve dogrulama verilerini yukle
            train_df = pd.read_csv(r"D:\Kairu_DS360_Projects\fifth_week_project\m5_forecasting\artifacts\datasets\train.csv")
            valid_df = pd.read_csv(r"D:\Kairu_DS360_Projects\fifth_week_project\m5_forecasting\artifacts\datasets\validation.csv")
            
            item_train = train_df[train_df['item_id'] == self.item_id]['sales'].copy()
            item_valid = valid_df[valid_df['item_id'] == self.item_id]['sales'].copy()
            
            self.train_series = item_train
            self.valid_series = item_valid
            
            print(f"Egitim Icin {len(self.train_series)} Gun Olusturuldu")
            print(f"Dogrulama Icin {len(self.valid_series)} Gun Olusturuldu")
            print(f"Egitim Icin Ortalama: {self.train_series.mean():.4f}")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Ilgili Dosya Yollari Bulunamadi...")
        
        except Exception as err:
            raise Exception(f"Zaman Serisi Olusturulurken Beklenmeyen Bir Hata Gerceklesti: {err}")
        
    def prepare_prophet_data(self):
        """Prophet Icin Gerekli Donusumleri Gerceklestirir"""
        print(f"Prophet Icin Uygun Format Donusumleri Gerceklestiriliyor...")
        prophet_train = pd.DataFrame(
            {
                'ds':self.train_series.index
                ,'y': self.train_series.values
            }
        )
        
        prophet_valid = pd.DataFrame(
            {
                'ds': self.valid_series.index
                ,'y': self.valid_series.values
            }
        )
        
        print(f"Prophet Modeli Icin Egitim Verisi: {len(prophet_train)} Satirdan Olusmaktadir")
        print(f"Prophet Modeli Icin Dogrulama Verisi: {len(prophet_valid)} Satirdan Olusmaktadir")
        print(f"Tarih Araliklari: {prophet_train['ds'].min()} - {prophet_train['ds'].max()}")
        
        return prophet_train, prophet_valid
        
        
    def train_prophet_model(self, prophet_train_df):
        """Prophet Model Egitimini Gerceklestirir"""
        print(f"Prophet Model Egitimi Gerceklestiriliyor...")
        try:
            # model parametreleri
            self.prophet_model = Prophet(
                daily_seasonality = True
                ,weekly_seasonality = True
                ,yearly_seasonality = False
                ,changepoint_prior_scale=0.05  # Trend degisim hassasiyet degeri
                ,seasonality_prior_scale=10.0  # Sezonluk hassasiyet degeri
                ,interval_width=0.95       # Guven araligi %95
                , n_changepoints=25         # Egilimin degistigi nokta  
            )
            ("Degerlendirme Icin Parametrelerin Aciklamasi Veriliyor...")
            print("Gunluk Sezonluk (daily_seasonality) : Acik")
            print("Haftalik Sezonluk (weekly_seasonality): Acik")
            print("Yillik Sezonluk (yearly_seasonality): Kapali")
        
            print("Model Egitimi Baslatiliyor...")
            self.prophet_model.fit(prophet_train_df)
        
            print("Prophet Model Egitimi Basariyla Gerceklestirildi.")
            
            # Model Bilesenlerini yazdiralim
            if hasattr(self.prophet_model , 'seasonalities'):
                print("Aktif Olarak Kullanilan Sezonluklar:")
                for name, seasonality in self.prophet_model.seasonalities.items():
                    print(f" {name} Isimli Parametrenin: Period Degeri={seasonality['period']}, Siralamasi={seasonality['fourier_order']}")
                    
        except Exception as err:
            raise Exception(f"Model Egitiminde Beklenmeyen Bir Hata Olustu: {err}")
        
        
    def make_prophet_forecast(self, forecast_steps = 28):
        """Prophet Modeli Kullanarak 28 Gunluk Tahmin Gerceklestir"""
        print(f"Prophet Modeli ile {forecast_steps} Gunluk Tahmin Gerceklestiriliyor...")
        
        try:
            # Gelecek icin bir cerceve olustur
            future_df = self.prophet_model.make_future_dataframe(periods = forecast_steps
                                                                 ,freq = 'D')
            print(f"Future Dataframe'in Kayit Sayisi: {len(future_df)}")
            print(f"Tahmin Araligi: {future_df['ds'].iloc[-forecast_steps]} - {future_df['ds'].iloc[-1]}")

            print("Tahmin Hesaplaniyor...")
            self.forecast_df = self.prophet_model.predict(future_df)
            
            # Son 28 gunu alalim
            forecast_period = self.forecast_df.tail(forecast_steps).copy()
            
            # Negatif deger varsa 0 ile degistirelim
            forecast_period['yhat'] = forecast_period['yhat'].clip(lower = 0)
            
            
            print(f"{forecast_steps} Gunluk Tahmin Gerceklestirildi")
            print(f"Ortalama Tahmin: {forecast_period['yhat'].mean():.4f}")
            print(f"Minimum Tahmin: {forecast_period['yhat'].min():.4f}")
            print(f"Maximum Tahmin: {forecast_period['yhat'].max():.4f}")
            
            # Guven araligi (confidence interval)
            ci_width = forecast_period['yhat_upper'].mean() - forecast_period['yhat_lower'].mean()
            print(f"Ortalama Guven Araligi Genisligi: {ci_width:.2f}")
            
            return forecast_period
        except Exception as err:
            raise Exception(f"Tahmin Asamasinda Beklenmeyen Bir Hata Gerceklesti: {err}")
        
    def calculate_metrics_and_compare_arima(self, forecast_period):
        """Prophet Modeli Icin Degerlendirme Metriklerini Hesaplar"""
        print("Prophet Modeli Icin Degerlendirme Metrikleri Hesaplaniyor...")
        
        # Gercek degerler
        y_true = self.valid_series.values
        # Tahmin degerleri 
        y_pred = forecast_period['yhat'].values[:len[y_true]] # ayni boyutta olmalilar.
        
        # Degerlendirme Metrikleri
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Mean Absolute Percentage Error
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = float('inf') # infinite => Sonsuz
        
        
        # Simetrik MAPE
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        
        mask = denominator != 0
        if mask.sum() > 0:
            smape = np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
        else:
            smape = float('inf') 
            
        #Metrikleri sozluk turunde kaydedelim
        self.metrics = {
            "MAE": mae
            ,'RMSE': rmse
            ,'MAPE': mape
            ,'sMAPE': smape
            ,'forecast_mean': y_pred.mean()
            ,'actual_mean': y_true.mean()
            
        }
        
        print("Degerlendirme Metrikleri Hesaplandi. Metrik Degerleri Gosteriliyor...")
        print(f"Ortalama Mutlak Hata (MAE): {mae:.4f}")
        print(f"Kok Ortalama Kare Hata (RMSE): {rmse:.4f}")
        print(f"Ortalama Mutlak Yuzde Hata (MAPE): {mape:.4f}")
        print(f"Simetrik Ortalama Mutlak Yuzde Hata (sMAPE): {smape:.4f}")
        
        print("ARIMA Model ile Prophet Model Degerleri Karsilastiriliyor...")
        if self.arima_metrics:
            print(f"   {'Metrik':<8} {'ARIMA':<10} {'Prophet':<10} {'Kazanan':<10}")
            print(f"   {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
        
        # Karsilastirilacak olan metrikler
        metrics_to_compare  = ['MAE', 'RMSE', 'sMAPE']
        for metric in metrics_to_compare:
            arima_value = self.arima_metrics[metric]
            prophet_value = self.metrics[metric]
            
            
            if metric == 'sMAPE' and (arima_value == float('inf') or prophet_value == float('inf')):
                winner = "N/A" # hesaplanamadigi icin kazanan bulunamaz
                
                arima_str = "∞" if arima_value == float('inf') else f"{arima_value:.4f}"
                prophet_str = "∞" if prophet_value == float('inf') else f"{prophet_value:.4f}"
            
            else:
                winner = "Prophet" if prophet_value > arima_value else "ARIMA"
                arima_str = f"{arima_value:.4f}"
                prophet_str = f"{prophet_value:.4f}"
            
            # Hizalayalim < anlami sola hizala
            print(f"   {metric:<8} {arima_str:<10} {prophet_str:<10} {winner:<10}")
            
        return self.metrics
    
    
    def calc_visualizations(self, forecast_period):
        """Prophet Modeli Icin Gerekli Gorsellestirmeleri Uretir"""
        print(f"Prophet Modeli Icin Gorsellestimeler Gerceklestiriliyor...")

        fig1 = self.prophet_model.plot(self.forecast_df
                                       ,figsize = (12, 8)
                                       )        
        
        axs = fig1.gca()
        valid_start = self.valid_series.index[0]
        axs.axvline(x = valid_start
                    ,color = "maroon"
                    ,linestyle = "--"
                    ,alpha = 0.7
                    ,label = "Dogrulama Baslangici")
        
        axs.legend()
        axs.set_title(f"{self.item_id} Urununun Prophet Tahmini", fontsize = 16, fontweight = 'bold', color = 'black')
        
        try:
            prophet_plot_path = f'{self.artifacts_path}/figures/prophet_{self.item_id}_components.png'
            fig1.savefig(prophet_plot_path, dpi=300, bbox_inches='tight')
            print(f"Bar Grafigi Kaydedildi: {prophet_plot_path}")
            plt.close()
            
        except FileNotFoundError:
            raise FileNotFoundError(f"{prophet_plot_path} Dosya Yolu Bulunamadi")
        
        except Exception as err:
            raise Exception(f"Bar Grafigi Kaydedilirken Beklenmeyen Bir Hata Olustu: {err}")
        
        fig , axes = plt.subplots(2, 2, figsize = (12, 8))
        
        # Ana tahmin grafikleri
        axs1 = axes[0, 0]
        # Son 100 gunu alalim
        train_plot = self.train_series.tail(100)
        
        axs1.plot(
            train_plot.index
            ,train_plot.values
            ,label = 'Egitim Verisi Gercek Degerleri'
            ,color = 'skyblue'
            ,linewidth = 1.5
        )
        
        axs1.plot(
            self.valid_series.index
            ,self.valid_series.values
            ,label = 'Dogrulama Verisi Gercek Degerleri'
            ,color = 'forestgreen'
            ,linewidth = 1.5
        )
        
        axs1.plot(
            forecast_period['ds']
            ,forecast_period['yhat']
            ,label = 'Prophet Tahmini'
            ,color = 'maroon'
            ,linewidth = 1.5
            ,linestyle = "--"
        )
        
        # Guven Araligi
        axs1.fill_between(
            forecast_period['ds']
            ,forecast_period['yhat_lower']
            ,forecast_period['yhat_upper']
            ,color = 'salmon'
            ,alpha = 0.3
            ,label = "Guven Araligi (%95)"
        )
        
        axs1.axvline(
            x = self.train_series.index[-1] #son degeri alir
            ,color = 'dimgray'
            ,linestyle = ':'
            ,alpha = 0.8
            ,label  ="Egitim/Dogrulama Ayrimi"
        )
        axs1.set_title(f'{self.item_id} Urunu Prophet Tahmini', fontsize = 16, fontweight = 'bold')
        axs1.set_ylabel('Satis Degeri')
        axs1.legend()
        axs1.grid(True, alpha = 0.4)
        
        # Model Karsilastirmasi
        axs2 = axes[0, 1] # 1.satir 2.sutun
        
        if self.arima_metrics:
            models = ['ARIMA', 'Prophet']
            
            mae_values = [self.arima_metrics['MAE']
                          ,self.metrics['MAE']]
            
            smape_values = [self.arima_metrics['sMAPE']
                            ,self.metrics['sMAPE']]
            
            
            x = np.arange(len(models))
            width = 0.4
            
            bars1 = axs2.bar(x - width/2, mae_values, width, label='MAE', alpha=0.7)
            bars2 = axs2.bar(x + width/2, smape_values, width, label='sMAPE (%)', alpha=0.7)
            
            axs2.set_title("ARIMA - Prophet Model Basarilari Karsilastirmasi", fontweight = 'bold')
            axs2.set_ylabel("Degerlendirme Metrik Degeri")
            axs2.set_xticks(x)
            axs2.set_xticklabels(models)
            axs2.legend()
            axs2.grid(True, alpha=0.2)
            
            
            # Degerleri sutun grafige yazdiralim
            for bar, value in zip(bars1, mae_values):
                axs2.text(bar.get_x() + bar.get_width()/2
                          ,bar.get_height() + 1
                        ,f'{value:.2f}'
                        ,ha='center'
                        ,va='bottom')
            
            for bar, value in zip(bars2, smape_values):
                axs2.text(bar.get_x() + bar.get_width() / 2
                          ,bar.get_height() + 1
                          ,f'{value:.2f}'
                          ,ha = 'center'
                          ,va = 'bottom')
                
            # Prophet modeli icin trend analizi
            axs3 = axes[1, 0] # 2. satir, 1.sutun
            axs3.plot(
                self.forecast_df['ds']
                ,self.forecast_df['trend']
                ,color = 'yellow'
                ,linewidth = 1.5
            )
            axs3.set_title("Prophet Trend Analizi", fontsize = 16, fontweight = 'bold')
            axs3.set_ylabel('Trend')
            axs3.grid(True, alpha = 0.4)
            
            
            
            axs4 = axes[1, 1]
            if 'weekly' in self.forecast_df.columns:
                axs4.plot(
                    self.forecast_df['ds']
                    ,self.forecast_df['weekly']
                    ,color = 'mediumpurple'
                    ,linewidth = 1.5
                    ,alpha = 0.6
                )
                
                axs4.set_title("Prophet Modeli Haftalik Mevsimsellik", fontsize = 16, fontweight = 'bold')
                axs4.set_ylabel("Haftalik Etki")
                axs4.grid(True, alpha = 0.4)
                
            else:
                axs4.text(
                    0.5
                    ,0.5
                    ,'Haftalik Mevsimsellik Bileseni Kullanilamiyor'
                    ,ha = 'center'
                    ,va = 'center'
                    ,transform=axs4.transAxes                    
                )
                axs4.set_title("Haftalik Mevsimsellik Etkisi (N/A)"
                               ,fontsize = 16
                               ,fontweight = 'bold')
                
            plt.suptitle(f"{self.item_id} Urunu Icin Prophet Modeli Analizi"
                         ,fontsize = 18
                         ,fontweight= 'bold')
            plt.tight_layout()
            
            plt.close()
            # Bunu kaydedelim
            try:
                main_plot_path = f'{self.artifacts_path}/figures/prophet_{self.item_id}_forecast.png'
                plt.savefig(main_plot_path, dpi=300, bbox_inches='tight')
                print(f" Prophe-ARIMA Model Karsilastirmasi: {main_plot_path}")
                plt.close()
            except FileNotFoundError:
                raise FileNotFoundError(f"{main_plot_path} Dosya Yolu Bulunamadi")
            
            except Exception as err:
                raise Exception(f"Grafik Beklenemeyen Bir Nedenle Kaydedilemedi: {err}")
            
    def save_results(self, forecast_period):
        """Prophet Modeli Sonuclarini Kaydeder"""
        print("Prophet Model Sonuclari Kaydediliyor...")
        model_info = {
            'item_id': self.item_id
            ,'model_type': 'Prophet'
            ,'params':
                {
                    'daily_seasonality': True
                    ,'weekly_seasonality': True
                    ,'yearly_seasonality': False
                    ,'changepoint_prior_scale': 0.05
                    ,'seasonality_prior_scale': 10.0
                    ,'interval_width': 0.95
                    ,'n_changepoints': 25
            }
        
            ,'train_period': f"{self.train_series.index.min()} to {self.train_series.index.max()}"
            ,'valid_period': f"{self.valid_series.index.min()} to {self.valid_series.index.max()}"
            ,'forecast_steps': len(forecast_period)
            ,'metrics': self.metrics
            ,'changepoints': self.prophet_model.changepoints.tolist() if hasattr(self.prophet_model, 'changepoints') else []
            ,'seasonalities': list(self.prophet_model.seasonalities.keys()) if hasattr(self.prophet_model, 'seasonalities') else []
            
        }
        
        try:
            model_path = f'{self.artifacts_path}/models/prophet_{self.item_id}.json'
            with open(model_path, 'w') as file:
                json.dump(model_info
                          ,file
                          ,indent = 4
                          ,default = str)

            print(f"Model Bilgileri {model_path} Dosya Yoluna Kaydedildi") 
        except Exception as err:
            print(f"Model Bilgileri Kaydedilirken Hata Olustu: {err} ")
        
        #Tahminleri kaydedelim
        try:
            forecast_save_df = pd.DataFrame(
            {
            'date': forecast_period['ds']
            ,'item_id': self.item_id
            ,'forecast': forecast_period['yhat']
            ,'forecast_lower': forecast_period['yhat_lower']
            ,'forecast_upper': forecast_period['yhat_upper']
            ,'actual': self.valid_series.values[:len(forecast_period)]
            })
        
        
            pred_path = f'{self.artifacts_path}/predictions/prophet_forecast_{self.item_id}.csv'
            forecast_save_df.to_csv(pred_path, index=False)
            print(f"Tahminler Kaydedildi: {pred_path}")
        except Exception as err:
            print(f"Tahminler Kaydedilirken Bir Hata Olustu: {err}")
            
        if self.arima_metrics:
            try:
                comparison = {
                    'item_id': self.item_id,
                    'comparison_date': datetime.now().isoformat(),
                    'arima_metrics': self.arima_metrics,
                    'prophet_metrics': self.metrics,
                    'winner_by_metric': {},
                    'summary': {
                        'arima_advantages': [
                            "Matematiksel olarak saglamdir",
                            "Duragan serilerde guclu",
                            "Parametre kontrolu yuksektir"
                    ],
                    'prophet_advantages': [
                        "Kolay kullanima sahiptir",
                        "Otomatik sezonluk yakalama mevcuttur",
                        "Tatil efektleri eklenebilir",
                        "Eksik veriye dayaniklidir",
                    ]
                }
                }
            
            # Metrik bazında kazanan
                for metric in ['MAE', 'RMSE', 'sMAPE']:
                    if metric in self.arima_metrics and metric in self.metrics:
                        arima_value = self.arima_metrics[metric]
                        prophet_value = self.metrics[metric]
                        
                        if arima_value == float('inf') or prophet_value == float('inf'):
                            comparison['winner_by_metric'][metric] = 'N/A'
                        else:
                            comparison['winner_by_metric'][metric] = 'Prophet' if prophet_value < arima_value else 'ARIMA'
                
            
                # Sonuclari Kaydedelim
                comparison_path = f'{self.artifacts_path}/predictions/arima_vs_prophet_{self.item_id}.json'
            
                with open(comparison_path, 'w') as file:
                    json.dump(comparison
                        ,file
                        ,indent=4,
                        default=str)
                print(f"Prophet-ARIMA Modellerinin Karsilastirma Raporu: {comparison_path}")
        
            except FileNotFoundError:
                print(f"Belirtilen Dosya Yollarinda Hata ile Karsilasildi!!!")
            except Exception as err:
                print(f"Kaydetme Sirasinda Hata Olustu: {err}")
        
              