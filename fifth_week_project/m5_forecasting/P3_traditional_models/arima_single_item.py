""" 
ARIMA (Aristmetic Integrated Moving Average) model tek bir urun icin satis tahmini yapar.
Bu model zaman serisi verilerinde trend ve mevsimsellik gibi ozellikleri yakalamak icin kullanilir.
"""

#%% Gerekli Kutuphaneleri Dahil Edelim (import required libraries)
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from itertools import product
from datetime import datetime, timedelta

import warnings
warnings.filterwarnings("ignore")

# statsmodels kutuphanesinden ARIMA modelini ve performans metriklerini import edelim
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# model degerledirme metrikleri (regresyon oldugu icin MSE, MAE)
from sklearn.metrics import mean_squared_error, mean_absolute_error


class ARIMAModelSingleItemForecaster:
    """ARIMA modeli kullanarak tek bir urun icin satis tahmini yapar."""
    
    
    def __init__(self, artifacts_path : str = './artifacts'):
        self.artifacts_path = os.path.abspath(artifacts_path)
        self.item_id = None
        self.ts_data = None
        self.train_series = None
        self.valid_series = None
        self.model = None
        self.forecast = None
        self.best_params = None
        self.metrics = {}
        self.create_artifacts_dirs()
        
        
    def create_artifacts_dirs(self):
        """Model artifact dizinlerini olusturur."""
        model_dir = os.path.join(self.artifacts_path, 'models')
        preds_dir = os.path.join(self.artifacts_path, 'predictions')
        figures_dir = os.path.join(self.artifacts_path, 'figures')
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(preds_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)
        """ 
        for sub_dir in [model_dir, predictions_dir, figures_dir]:
            os.makedirs(os.path.join(self.artifacts_path, sub_dir), exist_ok=True)
        """
    def select_top_item(self):
        """En cok satan urunu secer"""
        print("[INFO] En Cok Satan Urun Seciliyor...")
        try:
            # egitim verisini yukleyelim
            train_df = pd.read_csv(r"D:\Kairu_DS360_Projects\fifth_week_project\m5_forecasting\artifacts\datasets\train.csv"
                                   ,parse_dates= ['date']
                                   ,index_col= 'date')
            
            # urun bazinda toplam satislari hesaplayalim
            item_totals = train_df.groupby('item_id')['sales'].sum().sort_values(ascending=False)
            print("Urun Bazinda Toplam Satislar Hesaplandi.")
            for idx, (item, total) in enumerate(item_totals.items()):
                print(f"{idx+1}. Urun: {item}, Toplam Satis: {total}")
                
            # en cok satan urunu alalim
            self.item_id = item_totals.index[0]
            top_sales = item_totals.iloc[0]
            
            print(f"En Cok Satilan Urun: {self.item_id} \nToplam Satis Adedi: {top_sales}")
        except FileNotFoundError:
            raise FileNotFoundError("Egitim verisi bulunamadi. Lutfen create_m5_subset.py Kullanarak Veri Setini Hazirlayin.")

        return self.item_id
    
    def load_time_series(self):
        """Secilen urun icin zaman serisi verisini yukler."""
        print(f"[INFO] Zaman Serisi Verisi Yukleniyor: {self.item_id}")
        try:
            # egitim verisini yukleyelim
            train_df = pd.read_csv(r"D:\Kairu_DS360_Projects\fifth_week_project\m5_forecasting\artifacts\datasets\train.csv"
                                   ,parse_dates= ['date']
                                   ,index_col= 'date')
            valid_df = pd.read_csv(r"D:\Kairu_DS360_Projects\fifth_week_project\m5_forecasting\artifacts\datasets\validation.csv"
                                   ,parse_dates= ['date']
                                   ,index_col= 'date')
        # secilen urunun zaman serisi verisini alalim
        
            item_train = train_df[train_df['item_id'] == self.item_id]['sales'].copy()
            item_valid = valid_df[valid_df['item_id'] == self.item_id]['sales'].copy()
            
            print(f"Egitim Verisi Zaman Araligi: {item_train.index.min()} - {item_train.index.max()}")
            print(f"Dogrulama Verisi Zaman Araligi: {item_valid.index.min()} - {item_valid.index.max()}")
            print(f"Egitim Verisi Gun Sayisi: {len(item_train)}")
            print(f"Dogrulama Verisi Gun Sayisi: {len(item_valid)}")
            
            
            # Serilerimizi koruyalim
            self.train_series = item_train
            self.valid_series = item_valid
            
            # Verilerimizi birlestirelim
            self.ts_data = pd.concat([item_train, item_valid])
            
            print(f"[INFO] Toplam Gunluk Veri Sayisi: {len(self.ts_data)}")
            
            print("Temel Istatistik Bilgileri Getiriliyor...")
            print(f"Ortalama: {self.train_series.mean():.4f}")
            print(f"Standart Sapma: {self.train_series.std():.4f}")
            print(f"En Dusuk Satis: {self.train_series.min()}")
            print(f"En Yuksek Satis: {self.train_series.max()}")
            print(f"Hic Satis Yapilmayan Gun Sayisi: {(self.train_series == 0).sum()}")

        except FileNotFoundError:
            raise FileNotFoundError("Belirtilen Dosya Yollarinda Ilgili Klasorler Bulunamadi!!!")
        
        except Exception as err:
            raise Exception(f"Beklenmeyen Bir Hata Olustu: {err}")
               
        return self.ts_data
    
    
    def test_stationarity(self, series, title = 'Series'):
        """Stationarity (ADF) Test Uygular"""
      
        print(f"{title} Icin Stationarity Test Gerceklesitiriliyor...")
        
        adf_result = adfuller(series.dropna())
        print(f"ADF Istatistikleri: {adf_result[0]:.6f}")
        print(f"p-Degeri: {adf_result[1]:.6f}")
        print(f"Kritik Degerler: ")
        for key, value in adf_result[4].items():
            print(f"{key}: {value:.4f}")
            
        # Sonuc yorumu
        if adf_result[1] <= 0.05:
            print(f"Duragan (Stationary) p-Degeri < 0.05")
            is_stationary = True
        else:
            print(f"Duragan Degil (Non-Stationary) p-Degeri > 0.05")
            is_stationary = False
            
        # return {"is_stationary": is_stationary, "p_value": adf_result[1], "adf_stat": adf_result[0]}
        return is_stationary, adf_result[1]
    
    
    def determine_d_parameter(self):
        """d(Differencing order) parametresini belirle"""
        print("ARIMA Yontemi Icin d Parametresi Belirleniyor")
        
        series = self.train_series.copy()
        d = 0
        max_d =  2 # maksimum 2 kez 
        
        # orijinal serinin duraganligini test edelim
        is_stationary, p_value = self.test_stationarity(series, f'Orijinal: (d = {d})')
        
        # duragansa
        if is_stationary:
            print(f"{d} Parametre Ayari Gerekmiyor. Duragan...")
            return d
        
        
        for d in range(1, max_d + 1):
            diff_series = series.diff(d).dropna()
            
            if len(diff_series) < 50:
                print(f"d = {d} Degeri Icin Yeterli Veri Kalmamaktadir. d = {d - 1} Kullanilacaktir")
                return d - 1
            
            is_stationary, p_value = self.test_stationarity(diff_series,f"Ayarlanan d degeri: {d}")
            if is_stationary:
                print(f"d Degeri Olarak {d} Belirlendi")
                return d
        
        print(f"Optimal d Degeri Bulunamadi, d = 1 Olarak Belirlenecektir")
        return 1
    
    def grid_search_arima(self, max_p = 2, max_q = 2):
        """ARIMA Yontemi Icin Parametrelerin Grid Search Kullanilarak Optimize Edilmesi"""
        print("ARIMA Grid Search Islemi Gerceklestiriliyor...")
        
        d = self.determine_d_parameter()
        
        # grid search icin parametreler
        p_values = range(0, max_p + 1)
        q_values = range(0, max_q + 1)
        
        print(f"p Degerleri: {list(p_values)}")
        print(f"q Degerleri: {list(q_values)}")
        print(f"Kullanilan d Degeri: {d}")
        
        
        best_aic = float('inf')
        best_params = None
        results = [] 

        
        total_combinations = len(q_values) * len(p_values)
        current_combination = 0
        
        for p in p_values:
            for q in q_values:
                current_combination += 1
                try:
                    print(f"ARIMA Yontemi Icin ({p}, {d}, {q}) Parametreleri Deneniyor... ({current_combination}/{total_combinations}")
                    # modeli egitelim
                    model = ARIMA(self.train_series, order = (p, d, q)) # sira cok onemli, yoksa hata alirsin!!!
                    fitted_model = model.fit()
                    
                    aic = fitted_model.aic
                    bic = fitted_model.bic
                    
                    results.append(
                        {
                            'p':p
                            ,'d': d
                            ,'q': q
                            ,'AIC': aic
                            ,'BIC': bic
                            ,'converged': fitted_model.mle_retvals['converged'] if hasattr(fitted_model, 'mle_retvals') else True
                            
                        }
                    )
                    
                    if aic < best_aic:
                        best_aic = aic
                        best_params = (p, d, q)
                        print(f"En Iyi AIC = {aic:.4f}")
                except Exception as err:
                    print(f"Beklenmeyen Bir Hata Olustu: {str(err)[:50]}")
                    results.append(
                        {
                            'p': p
                            ,'q': q
                            ,'d' : d
                            ,'AIC': np.inf
                            ,'BIC': np.inf
                            ,'converged': False
                            
                        }
                    )
        # DataFrame'e donusturelim
        results_df = pd.DataFrame(results)
        print("Grid Search Sonuclari Gosteriliyor...")
        print(f"En Iyi Parametreler: ARIMA{best_params}")
        print(f"En Iyi AIC Degeri: {best_aic:.4f}")
        
        # En iyi 3 sonuc
        top_results = results_df[results_df['converged']].nsmallest(3, 'AIC')
        print("En Iyi 3 Model")
        
        
        for idx, (_, row) in enumerate(top_results.iterrows(), 1):
            print(f"{idx}. ARIMA({int(row['p'])},{int(row['d'])},{int(row['q'])}) - AIC: {row['AIC']:.2f}")
        
        self.best_params = best_params
        return best_params, results_df
    
    
    def train_arima_model(self):
        """ARIMA Modelini Egitir"""
        print(f"En Iyi Parametreler Kullanilarak ARIMA{self.best_params} Modeli Egitiliyor...")
        try:
            # modeli egitelim
            model = ARIMA(self.train_series, order = self.best_params)
            self.model = model.fit()
            print("Model Egitimi Basariyla Gerceklestirildi")
            print(f"AIC: {self.model.aic:.4f}")
            print(f"BIC: {self.model.bic:.4f}")
            print(f"Log Likelihood Yontemiyle: {self.model.llf:.4f}")
            
            print("Model Parametreleri")
            if hasattr(self.model, 'params'):
                for param_name, param_value in self.model.params.items():
                    print(f"{param_name} parametresinin degeri: {param_value}")
        except Exception as err:
            raise Exception(f"Model Egitimi Sirasinda Beklenmeyen Bir Hata Olustu: {err}")
    
    def make_forecast(self, forecast_steps = 28):
        """28 Gunluk Tahmin Gerceklestirir"""
        print(f"{forecast_steps} Gunluk Tahmin Basliyor...")
        try:
            forecast_result = self.model.forecast(steps = forecast_steps)
            
            # guven araliginda calis
            try:
                forecast_ci = self.model.get_forecast(steps = forecast_steps).conf_int()
                forecast_lower = forecast_ci.iloc[:, 0]
                forecast_upper = forecast_ci.iloc[:, 1]
            except:
                forecast_lower = None
                forecast_upper = None 
            
            # Tarih araligi olusturalim
            last_date = self.train_series.index[-1]
            forecast_dates = pd.date_range(start = last_date + timedelta(days = 1)
                                           ,periods = forecast_steps
                                           ,freq= 'D')
            
            # Tahmin gerceklestirelim
            self.forecast = pd.Series(forecast_result, index = forecast_dates)
            
            # Eger negatif deger varsa 0 yap
            self.forecast = self.forecast.clip(lower = 0)
            
            print("Tahminlerle Ilgili Ozet Istatistikler Getiriliyor...")
            print(f"{forecast_steps} Gunluk Tahmin Gerceklestirildi")
            print(f"Tahmin Araliklari: {self.forecast.index.min()} - {self.forecast.index.max()}")
            print(f"Ortalama Tahmin Degeri: {self.forecast.mean():.4f}")
            print(f"Minimum Tahmin Degeri: {self.forecast.min():.4f}")
            print(f"Maksimum Tahmin Degeri: {self.forecast.max():.4f}")
            
            return self.forecast, forecast_lower, forecast_upper
        
        except Exception as err:
            raise Exception(f"Tahminleme Sirasinda Beklenmeyen Hata Olustu: {err}")
        
    
    
    def calculate_metrics(self):
        """MAE, MSE Degerlerini Hesaplar"""
        print("Tahmi Metrikleri Hesaplaniyor...")
        
        
        # Gercek ve Tahmin Degerleri
        y_true = self.valid_series.values
        y_pred = self.forecast.values
        
        # Uzunluklarini esitlememiz gerekmektedir.
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        # Metrikler (Mean Absolute Error - Root Mean Square Error)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        
        # MAPE (Mean Absolute Percentage Error)
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = float("inf")
        
        
        #sMAPE
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask = denominator != 0
        if mask.sum() > 0:
            smape = np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
        else:
            smape = float('inf')
        
        self.metrics = {
            "MAE": mae
            ,"RMSE": rmse
            ,"MAPE": mape
            ,"sMAPE": smape
        }
        
        print("Tahmin Degerlendirmesi Gerceklestiriliyor...")
        print(f"Ortalama Mutlak Hata (MAE): {mae:.4f}")
        print(f"Kok Ortalama Kare Hata (RMSE): {rmse:.4f}")
        print(f"Ortalama Mutlak Yuzde Hata (MAPE): {mape:.4f}")
        print(f"Simetrik Ortalama Mutlak Yuzde Hata (sMAPE): {smape:.4f}")
        
        return self.metrics
    
    
    def create_visualizations(self):
        """Gorsellestirmeleri Gerceklestirir"""
        print("Gorsellestirme Islemi Baslatiliyor...")
    
        fig, axes = plt.subplots(2, 2, figsize = (12, 8))
        
        # Ana tahmin
        axs1 = axes[0, 0]
        train_plot = self.train_series.tail(100)
        
        axs1.plot(train_plot.index
                  ,train_plot.values
                  ,label = 'Gercek Egitim Verileri'
                  ,color = 'skyblue'
                  ,linewidth = 1.5)
        
        axs1.plot(self.valid_series.index
                  ,self.valid_series.values
                  ,label = 'Gercek Dogrulama Verileri'
                  ,color = 'forestgreen'
                  ,linewidth = 1.5)
        
        axs1.plot(self.forecast.index
                  ,self.forecast.values
                  ,label = "ARIMA Model Tahmini"
                  ,color = "maroon"
                  ,linewidth = 1.5
                  ,linestyle = "--")
        
        
        axs1.axvline(x=self.train_series.index[-1], color='gray', linestyle=':', alpha=0.6, 
                   label='Egitim ve Dogrulama Ayrimi')
        
        axs1.set_title(f'ID Degeri {self.item_id} - ARIMA{self.best_params} Tahmini', fontweight='bold')
        axs1.set_ylabel('Satislar')
        axs1.legend()
        axs1.grid(True, alpha=0.4)
        
        
        # 2. Residuals (artıklar)
        axs2 = axes[0, 1]
        residuals = self.model.resid
        axs2.plot(
            residuals.index
            ,residuals.values
            ,color = 'orange'
            ,alpha = 0.6
        )
        axs2.axhline(
            y = 0
            ,color = 'black'
            ,linestyle = '-'
            ,alpha = 0.7
        )
        axs2.set_title("Residuals (Artiklar)", fontsize = 16, fontweight = 'bold')
        axs2.set_ylabel("Artik Degeri")
        axs2.grid(True, alpha = 0.2)
        
        # Artik degerleri icin ACF
        axs3 = axes[1, 0]
        try:
            plot_acf(residuals.dropna()
                     ,ax = axs3
                     ,lags = 20
                     ,alpha = 0.04)
            axs3.set_title("Artik Degerler Icin ACF", fontsize = 16, color = 'black', fontweight = 'bold')
        except:
            axs3.text(0.5
                      ,0.5
                      ,'Deger Hesaplanamadi'
                      ,ha = 'center'
                      ,va = 'center'
                      ,transform = axs3.transAxes)
            axs3.set_title("Artiklar Icin ACF Hesaplama Hatasi"
                           ,color = 'maroon'
                           ,fontweight = 'bold')
        
        #Artiklar icin PACF 
        axs4 = axes[1, 1]
        try:
            plot_pacf(residuals.dropna()
                      ,ax = axs4
                      ,lags = 20
                      ,alpha = 0.04)
            axs4.set_title("Artiklar Icin PACF", fontweight = 'bold')
        except:
            axs4.text(
                0.5
                ,0.5
                ,'PACF Hesaplanamadi'
                ,ha = 'center'
                ,va = 'center'
                ,transform = axs4.transAxes
            )
            axs4.set_title('Artiklar Icin PACF Hesaplamasinda Hata Olustu'
                           ,color = 'maroon'
                           ,fontweight = 'bold')
            
            plt.tight_layout()
            
            # Kaydedelim
            figure_path = f'{self.artifacts_path}/figures/arima_{self.item_id}_forecast.png'
            plt.savefig(figure_path, dpi = 300, bbox_inches = 'tight')
            print(f"Tahmin Grafikleri Kaydedildi: {figure_path} Dosya Yolunda Bulabilirsiniz...")
            plt.close()
            
            
            # Metrikleri gorsellestirelim
            fig, axs = plt.subplots(1, 1, figsize = (10, 8))
            
            metrics_names =  ['MAE','RMSE', 'sMAPE (%)']
            metrics_values = [self.metrics['MAE'], self.metrics['RMSE'], self.metrics['sMAPE']]
            
            bars = axs.bar(metrics_names
                           ,metrics_values
                           ,color=['darkblue', 'coral', 'forestgreen'])
            
            
            # Bar grafik uzerine yazdiralim
            for bar, value in zip(bars, metrics_values):
                axs.text(
                    bar.get_x() + bar.get_width() / 2
                    ,bar.get_height() + 0.1
                    ,f'{value:.4f}'
                    ,ha = 'center'
                    ,va = 'bottom'
                    ,fontweight = 'bold'
                    ,color = 'black'
                )            
            axs.set_title(f"{self.item_id} ID'li ARIMA{self.best_params} Icin Performans Metrikleri"
                          ,fontsize = 16
                          ,fontweight = 'bold')
            axs.set_ylabel('Degerlendirme Metrik Degerleri')
            axs.grid(True, alpha = 0.2)
            plt.tight_layout()
            
            # Bunlari klasore kaydedelim
            metrics_path = f'{self.artifacts_path}/figures/arima_{self.item_id}_metrics.png'
            plt.savefig(metrics_path, dpi = 300, bbox_inches = 'tight')
            print(f"Metrik Grafikleri Kaydedildi. Goz Atmak Icin: {metrics_path}")
            plt.close()
            
            
    def save_results(self):
        """Sonuclari Kaydeder"""
        print("Model Sonuclariniz Models Klasorune Kaydediliyor...")
        model_path = f'{self.artifacts_path}/models/arima_{self.item_id}.pkl'
            
        with open(model_path, 'wb') as file:
            pickle.dump(
                {
                    'model':self.model
                    ,'item_id': self.item_id
                    ,'best_params': self.best_params
                    ,'train_series': self.train_series
                    ,'metrics': self.metrics
                }
                ,file 
            )
        print(f"Modeller {model_path} Dosya Yoluna Kaydedildi")
            
        # Tahminleri kaydet
        forecast_df = pd.DataFrame(
            {
                'date': self.forecast.index
                ,'item_id': self.item_id
                ,'forecast': self.forecast.values
                ,'actual': self.valid_series.values[:len(self.forecast)]
            }
        )
            
        pred_path = f'{self.artifacts_path}/predictions/arima_forecast_{self.item_id}.csv'
        forecast_df.to_csv(pred_path, index = False)
        print(f"Tahminler {pred_path} Dosya Yoluna Kaydedildi")
            
            
        # Ozet rapor olusturalim
        report = {
            'item_id': self.item_id
            ,'model_type': 'ARIMA'
            ,'params': self.best_params
            ,'train_period': f"{self.train_series.index.min()} to {self.train_series.index.max()}"
            ,'valid_period': f"{self.valid_series.index.min()} to {self.valid_series.index.max()}"
            ,'forecast_steps': len(self.forecast)
            ,'metrics': self.metrics
            ,'model_aic': self.model.aic
            ,'model_bic': self.model.bic
        }
            
        import json
        report_path = f"{self.artifacts_path}/predictions/arima_report_{self.item_id}.json"
        with open(report_path, 'w') as file:
            json.dump(
                report
                ,file
                ,indent = 4
                ,default= str
            )
        print(f"Ozet Rapor Olusturuldu. {report_path} Dosya Yolundan Goz Atabilirsiniz")
            
    # Tek bir cati altinda toplayalim
    def run_arima_pipeline(self):
        """ARIMA Modeli Icin Pipeline Uygular"""
        try:
            # En yuksek satisa sahip urunu getir
            self.select_top_item()
            
            # Zaman serisi olustur
            self.load_time_series()
            
            # Parametre optimizasyonu gerceklestir
            best_params, grid_results = self.grid_search_arima()
            
            
            # Modeli egitelim
            self.train_arima_model()
            
            
            # Tahmin yaptiralim
            self.make_forecast()
            
            # Gercek deger ile tahmin degerleri arasi farklari degerlendirelim
            self.calculate_metrics()
            
            
            # Sonuclari gorsellestirelim
            self.create_visualizations()
            
            # Metrik ve gorsellestirme sonuclarini kaydedelim
            self.save_results()
            
            print(f"ARIMA Modeli Icin Tahminleme Gerceklestirildi")
            print("Hizli Goz Atis Gerceklestiriliyor...")
            print(f"Kullanilan Model Ismi: ARIMA{self.best_params}")
            print(f"Kullanilan Urun: {self.item_id}")
            print(f"Simetrik Ortalama Mutlak Yuzde Hata: {self.metrics['sMAPE']:.4f}%")
            print(f"Ciktilarin Kaydedildigi Dosya Yolu: {self.artifacts_path}/")
            return  self.model, self.forecast, self.metrics
            
        except Exception as err:
            raise Exception(f"Pipeline Sirasinda Beklenmeyen Hata Olustu: {err}")


def main():
    """Ana Fonksiyonumuz"""
    print("ARIMA Modeli Kullanilarak Tek Urun Satis Tahmini Gerceklestiriliyor...")
    try:
        forecaster = ARIMAModelSingleItemForecaster()
        # model icin adimlari tek bir catida calistir
        model, forecast, metrics = forecaster.run_arima_pipeline()
        print("Tek Urun Tahminlemesi Basariyla Gerceklestirildi")
    except KeyboardInterrupt:
        print(f"Uygulama Kullanici Tarafindan Durduruldu")
    
    except Exception as err:
        raise Exception(f"ARIMA Tahminlemesinde Beklenmeyen Hata Olustu: {err} ")
    
if __name__ == '__main__':
    main()