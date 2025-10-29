#%% Gerekli kutuphaneleri import et (import required libraries)
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime



""" 
Onemli Notlar:
Lag: Gecmis satis degerlerinin gelecekteki satislari tahmin etmek icin kullanilmasi
Rolling: Kisa donemli hareketli ortalamalar veya toplamlari hesaplamak icin kullanilir (Gurultuyu azaltmaktadir)
Tarih: Mevsimsellik, tatiller ve ozel gunler gibi tarihsel ozellikleri yakalamak icin kullanilir
"""

def create_features():
    print("Veri Seti Icin Lag ve Rolling Ozellikleri Olusturuluyor...")
    try:
        if not os.path.exists(r'D:\Kairu_DS360_Projects\fifth_week_project\m5_forecasting\artifacts\datasets'):
            print(f"./artifacts/datasets dizini bulunamadi. Lutfen once veri setini hazirlayin.")
            print('create_sample_data.py dosyasini calistirin.')
            return None, None, None, None, None, None
        else:
            print("Isleminize Devam Ediliyor...")
    except Exception as err:
        print(f"Hata olustu: {err}")
        return None, None, None, None, None, None
    
    
    print("Egitim ve Dogrulama Verileri Yukleniyor...")
    
    try:
        train_df = pd.read_csv(r'D:\Kairu_DS360_Projects\fifth_week_project\m5_forecasting\artifacts\datasets\train.csv', parse_dates=['date']
                               , index_col= 'date')
        print(f"Egitim DataFrame'inin Boyutu: {train_df.shape}")
        
        valid_df = pd.read_csv(r'D:\Kairu_DS360_Projects\fifth_week_project\m5_forecasting\artifacts\datasets\valid.csv', parse_dates=['date'])
        print(f"Dogrulama DataFrame'inin Boyutu: {valid_df.shape}")
        
        
        # Veri tip kontrolu gerceklestirelim
        print(f"Egitim DataFrame'inin Tarih Araligi: {train_df.index.min()} - {train_df.index.max()}")
        print(f"Dogrulama DataFrame'inin Tarih Araligi: {valid_df.index.min()} - {valid_df.index.max()}")
        
         
    except FileNotFoundError:
        print("Veri Dosyasi Bulunamadi: {err}")
        print("Sorunu Gidermek Icin create_sample_data.py Dosyasini Calistirin.")
        return None, None, None, None, None, None
    
    
    print("Egitim ve Dogrulama Veri Setleri Birlestiriliyor...")

    all_df = pd.concat([train_df, valid_df]).sort_index()
    print(f"Birlestirilmis DataFrame'in Boyutu: {all_df.shape}")
    print(f"Birlestirilmis DataFrame'in Tarih Araligi: {all_df.index.min()} - {all_df.index.max()}")
    
    
    # Her urune ozel oznitelik muhendisligi gerceklestirelim
    print("Oznitelik Muhendisligi(Feature Engineering) Gerceklestiriliyor...")
    feature_data =[]
    
    for item_id in all_df['item_id'].unique():
        print(f"{item_id} ID'li Urun Isleniyor...")
        
        item_df = all_df[all_df['item_id'] == item_id].copy()
        item_df = item_df.sort_index() # Tarih siralamasi oldukca onemlidir.
        
        ########################### LAG Ozellikleri ########################
        print("Lag Ozellikleri Ekleniyor...")
        item_df['lag_1'] = item_df['sales'].shift(1) #1 gun onceki satislar
        item_df['lag_7'] = item_df['sales'].shift(7) # 1 hafta onceki 
        item_df['lag_28'] = item_df['sales'].shift(28) # 4 hafta onceki => 1 hafta 7 gun
        
        
        ########################### Rolling Ozellikleri ########################
        # Hareketli ortalmalarin gucunden yararlanalim
        print("Rolling Ozellikleri Ekleniyor...")
        item_df['rolling_mean_7'] = item_df['sales'].rolling(window = 7, min_periods = 1).mean()
        item_df['rolling_mean_28'] = item_df['sales'].rolling(window = 28, min_periods = 1).mean()
        
        
        ########################### Tarih Ozellikleri ########################
        # Oruntu, mevsimsellik vb. gibi ozellikleri yakalamak icin onemlidir.
        print("Tarih Ozellikleri Ekleniyor...")
        item_df['dow'] = item_df.index.dayofweek # 0-6 arasi (Pazartesi, Sali .... Pazar)
        item_df['dom'] = item_df.index.day # day of month ayin kacinci gunu 1 - 31
        item_df['weekofyear'] = item_df.index.isocalendar().week # 1-53
        item_df['month'] = item_df.index.month # 1-12
        
        
        # urun ve magaza bilgilerini kaybetmemeliyiz, bilgi sizintisi riski olabilir ancak raporlama icin onemli
        item_df['store_id'] = item_df['store_id'].iloc[0]
        item_df['item_id'] = item_id
        
        feature_data.append(item_df)
        
        
    feature_df = pd.concat(feature_data, ignore_index= False)
    feature_df = feature_df.sort_index()
    print(f"Oznitelik Muhendisligi Asamasi Tamamlandi. Oznitelik Cikarimi Gerceklestirilen DataFrame'in Boyutu: {feature_df.shape}")
        
        
    # Eksik degerler var mi kontrol et ve onlari doldur
    print("feature_df Icerisindeki Eksik Degerler Kontrol Ediliyor...")
    nan_counts = feature_df.isnull().sum()
    nan_features = nan_counts[nan_counts > 0]
    
    
    if len(nan_features) > 0:
        print("Eksik Degere Sahip Oznitelikler")
        for feature, count in nan_features.items():
            print(f"{feature} degiskeninin eksik deger sayisi: {count}\n Orani Ise: ({count/len(feature_df)*100:.2f}")
    else:
        print("Herhangi Bir Eksik Deger Bulunamadi.")
        
    
    # Eksik degerleri doldur
    lag_features = ['lag_1', 'lag_7', 'lag_28']
    for lag_col in lag_features:
        if lag_col in feature_df.columns:
            before_count = feature_df[lag_col].isnull().sum()
            feature_df[lag_col] = feature_df[lag_col].fillna(0)
            after_count = feature_df[lag_col].isnull().sum()
            print(f"Onceki Durum: {lag_col}: {before_count} => Sonraki Durum: {after_count} Eksik Deger")
    
    # roll oznitelikleri icin
    roll_features = ['rolling_mean_7', 'rolling_mean_28']
    for roll_col in roll_features:
        if roll_col in feature_df.columns:
            before_count = feature_df[roll_col].isnull().sum()
            if before_count > 0: # eksik veri varsa
                feature_df[roll_col] = feature_df[roll_col].fillna(method = 'ffill').fillna(0)
                after_count = feature_df[roll_col].isnull().sum()
                print(f"Onceki Durumdaki Eksik Deger Sayisi: {roll_col}: {before_count}\n Sonraki Durumdaki Eksik Deger Sayisi: {after_count}")
    
    # Toplamda eksik veri kaldi mi?
    final_nan = feature_df.isnull().sum().sum()
    print(f"feature_df DataFrame'indeki Toplam Eksik Deger Sayisi: {final_nan}")
    
    # Dogrulama ve egitim verilerinin baslangic ve bitisi onemlidir
    train_end_date = train_df.index.max()
    valid_start_date = valid_df.index.min()
    
    print(f"Egitim Verisinin Son Tarih Bilgisi: {train_end_date}")
    print(f"Dogrulama Verisinin Baslangic Tarih Bilgisi: {valid_start_date}")
    
    
    # oznitelikleri train ve valid olarak bolelim
    fe_train = feature_df[feature_df.index <= train_end_date].copy()
    fe_valid = feature_df[feature_df.index >= valid_start_date].copy()
    
    print(f"Oznitelik Muhendisligi Uygunlanmis DataFrame'in Egitim Veri Seti Boyutu: {fe_train.shape}")
    print(f"Oznitelik Muhendisligi Uygunlanmis DataFrame'in Dogrulama Veri Seti Boyutu: {fe_valid.shape}")
    
    # Hedef degisken ile oznitelikleri ayiralim X ve y ayrimini gerceklestirelim
    print("Hedef Degisken (y) ile Diger Oznitelikler (X) Ayrimi Gerceklestiriliyor...")
    
    target_col = 'sales'
    feature_cols = [col for col in fe_train.columns if col not in [target_col, 'item_id','store_id']]
    
    print(f"Oznitelik (X) Sayisi: {len(feature_cols)}")
    print(f"X: {feature_cols}")
    
    
    # Egitim ve Dogrulama olarak bol
    X_train = fe_train[feature_cols].copy()
    y_train = fe_train[target_col].copy()
    
    
    X_valid = fe_valid[feature_cols].copy()
    y_valid = fe_valid[target_col].copy()
    
    print("Egitim ve Dogrulama Seti Ayrilan Kayitlarin Sayisi Gosteriliyor...")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_valid: {X_valid.shape}, y_valid: {y_valid.shape}")
    
    
    # Sizinti riski tasiyan magaza ve urun id bilgilerini raporlamak icin sakla
    train_metadata = fe_train[['store_id','item_id']].copy()
    valid_metadata = fe_valid[['store_id','item_id']].copy()
    
    # parquet formatinda kaydet
    print("Oznitelik Muhendisliginden Gecirilmis Veriler Kaydediliyor...")
    fe_train_path = r'D:\Kairu_DS360_Projects\fifth_week_project\m5_forecasting\artifacts\datasets\fe_train.parquet'
    fe_valid_path = r'D:\Kairu_DS360_Projects\fifth_week_project\m5_forecasting\artifacts\datasets\fe_valid.parquet'
    
    
    fe_train.to_parquet(fe_train_path)
    fe_valid.to_parquet(fe_valid_path)
    
    print(f"fe_train Dosya Yolu: {fe_train_path}")
    print(f"fe_valid Dosya Yolu: {fe_valid_path}")
    
    X_train.to_parquet(r"D:\Kairu_DS360_Projects\fifth_week_project\m5_forecasting\artifacts\datasets\X_train.parquet")
    y_train.to_parquet(r"D:\Kairu_DS360_Projects\fifth_week_project\m5_forecasting\artifacts\datasets\y_train.parquet")
    
    X_valid.to_parquet(r"D:\Kairu_DS360_Projects\fifth_week_project\m5_forecasting\artifacts\datasets\X_valid.parquet")    
    y_valid.to_parquet(r"D:\Kairu_DS360_Projects\fifth_week_project\m5_forecasting\artifacts\datasets\y_valid.parquet")    
    
    
    print("X_train, X_valid, y_train, y_valid Dosyalari Kaydedildi...")
    
    
    
    # Ozellik analizi yapalim
    print("Ozelliklerin Istatistiksel Bilgileri")
    feature_stats = X_train.describe()
    print(f"Sayisal Degiskenlerin Istatistikleri: {feature_stats.round(2)}")
    
    
    # Korelasyon iliskisi
    corr_with_target = X_train.corrwith(y_train).sort_values(ascending = False)
    
    print("En Yuksek Iliskiye Sahip Oznitelikler")
    for feature, corr in corr_with_target.items():
        print(f"{feature: 15}: {corr:6.4f}")
        
    # Gorsellestirme
    print("Ozniteliklerin Dagilim Grafikleri Olusturuluyor...")
    n_features = len(feature_cols)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize = (15, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_features == 1 else axes # duzlestirme yapiliyor, tek katmana indirgeme
    
    
    for idx, feature in enumerate(feature_cols):
        if idx < len(axes):
            ax = axes[idx]
            # Histogram ile dagilimi goster
            X_train[feature].hist(bins = 30, alpha = 0.6, ax = ax)
            ax.set_title(f"{feature}\Ortalamasi: {X_train[feature].mean():.4f}")
            ax.set_ylabel("Frekans")
            ax.grid(True, alpha = 0.4)
            
    
    # bos alt grafikleri saklayalim
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Egitim Setinin Oznitelik Dagilimi', fontsize = 16, fontweight ='bold', color = 'black')
    plt.tight_layout()
    
    
    # Grafigi kaydedelim
    hist_path = r"D:\Kairu_DS360_Projects\fifth_week_project\m5_forecasting\artifacts\figures\feature_distributions.png"
    try:
        plt.savefig(hist_path, dpi = 300, bbox_inches = 'tight')
        print(f"Histogram Dagilimlari Kaydedildi. Dosya Yolu: {hist_path}")
    except FileNotFoundError:
        print(f"{hist_path} Dosya Yolu Bulunamadi.")
    
    plt.close()
    
    
    # Korelasyon matrisini cizdirelim
    plt.figure(figsize = (12, 10))
    correlation_matrix = X_train.corr()
    
    # heatmap
    mask = np.triu(correlation_matrix)
    sns.heatmap(
        correlation_matrix
        ,mask = mask
        ,annot = True
        ,cmap = 'coolwarm'
        ,center = 0
        ,square = True
        ,fmt = '.1f'
        ,cbar_kws= {'label': 'Korelasyon'}

    )
    plt.title("Ozniteliklerin Korelasyon Matrisi", fontsize = 14, fontweight = 'bold', color = 'dimgray')
    plt.tight_layout()
    
    corr_path = r"D:\Kairu_DS360_Projects\fifth_week_project\m5_forecasting\artifacts\figures\correlations.png"
    try:
        save_dir = os.path.dirname(corr_path)
        os.makedirs(save_dir, exist_ok=True)

        plt.savefig(corr_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Korelasyon matrisi başarıyla kaydedildi: {corr_path}")

    except Exception as err:
        print(f"Beklenmeyen Bir Hata Olustu: {err}")
    plt.close()
    
    
    # 
    print(f"Toplam Ozellik Sayisi: {len(feature_cols)}")
    print(f"Lag Ozellikleri: {len([feat for feat in feature_cols if 'lag' in feat])}")
    print(f"Rolling özellikleri: {len([feat for feat in feature_cols if 'roll' in feat])}")
    print(f"Tarih Ozellikleri: {len([feat for feat in feature_cols if feat in ['dow', 'dom', 'weekofyear', 'month']])}")
    print(f"Egitim Ornekleri: {len(X_train):,}")
    print(f"Dogrulama Ornekleri: {len(X_valid):,}")
    print(f"Hedef Ortalamasi (Egitim Verisi): {y_train.mean():.2f}")
    print(f"Hedef Standart Sapma (Egitim Verisi): {y_train.std():.2f}")
    
    print(f"En Onemli 5 Iliski (Korelasyon Bazinda):")
    top_features = corr_with_target.abs().nlargest(5)
    for idx, (feature, corr) in enumerate(top_features.items(), 1):
        print(f"  {idx}. {feature}: {corr:.3f}")
    
    print(f"Oznitelik Muhendisligi (Feature Engineering) Tamamlandi!")
    
    
    return fe_train, fe_valid, X_train, y_train, X_valid, y_valid


def save_figures(figure_path: str):
    """Grafikleri belirtilen yola kaydeder."""
    try:
        os.makedirs(os.path.dirname(figure_path), exist_ok=True)
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        print(f"Grafik başarıyla kaydedildi: {figure_path}")
    except Exception as err:
        print(f"Grafik kaydedilirken hata oluştu: {err}")


def main():
    result = create_features()
    
    if result is None or (isinstance(result, tuple) and result[0] is None):
        print("Oznitelik Muhendisligi Basarisizlikla Sonuclandi. Modul Olarak P1'i Calistirin!!!")
        return False
    else:
        print("Oznitelik Muhendisligi (Feature Engineering) Basarili Bir Sekilde Gerceklestirildi")
        return True
    
if __name__ == '__main__':
    try:
        print("M5 Veri Seti Icin Oznitelik Muhendisligi Baslatiliyor...")
        result = create_features()
        if result is None or (isinstance(result, tuple) and result[0] is None):
            print("Oznitelik Muhendisligi Basarisizlikla Sonuclandi")
        
        else:
            fe_train, fe_valid, X_train, y_train, X_valid, y_valid = result
            print("Oznitelik Muhendisligi Basariyla Gerceklesi") 
    except Exception as err:
        print(f"Beklenmeyen Bir Hata Olustu: {err}")
        import traceback
        print("Cikis Yapiliyor...")
        traceback.print_exc()
        