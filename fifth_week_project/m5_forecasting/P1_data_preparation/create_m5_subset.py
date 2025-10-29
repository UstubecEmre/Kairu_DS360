#%% import required libraries (Gerekli kutuphaneleri dahil et)
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

#%% Set working directory (Calisma dizinini ayarla)
def ensure_dir_exists(dir_path: str):
    p = Path(dir_path)
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
        print(f"Klasor olusturuldu: {dir_path}")
    else:
        print(f"Klasor zaten mevcut: {dir_path}") 


def create_m5_subset():
    try:
        if not Path('./artifacts/datasets').exists():
            os.makedirs('./artifacts/datasets', exist_ok= True)
            print("Klasor olusturuldu: ./artifacts/datasets")
            #print("Created directory: ./artifacts/datasets")
        else:
            #print("Directory already exists: ./artifacts/datasets")
            print("Klasor zaten mevcut: ./artifacts/datasets")
        if not Path('./artifacts/figures').exists():
            os.makedirs('./artifacts/figures', exist_ok= True)
            #print("Created directory: ./artifacts/figures")
            print("Klasor olusturuldu: ./artifacts/figures")
        else:
            #print("Directory already exists: ./artifacts/figures")
            print("Klasor zaten mevcut: ./artifacts/figures")
    except Exception as err:
        #print(f"Error creating directories: {err}")
        sys.exit(1)
        
    # veya tek elden klasor olusturma fonksiyonu (tum klasorleri olusturmak icin)
    
    
        # read datasets (Verisetlerinin okunmasi)
    try:
        #print("Reading sales_train_validation.csv...")
        print("sales_train_validation.csv okunuyor...")
        sales_train_val_df = pd.read_csv(r'D:\Datasets\sales_train_validation.csv')
        print(f"sales_train_validation.csv boyut bilgileri: {sales_train_val_df.shape}")
        #print("Sales train validation data shape:", sales_train_val_df.shape)
            
        #print("Reading calendar.csv...")
        print("calendar.csv okunuyor...")
        calendar_df = pd.read_csv(r"D:\Datasets\calendar.csv")
            
        # convert to datetime (tarih formatina cevirme)
        calendar_df['date'] = pd.to_datetime(calendar_df['date'])
        #print(f"Calendar data shape: {calendar_df.shape}")
        print("calendar.csv boyut bilgileri:", calendar_df.shape)
        try:
            #print("Reading sell_prices.csv...")
            print("sell_prices.csv okunuyor...")
            prices_df = pd.read_csv(r"D:\Datasets\sell_prices.csv")
            #print(f"Prices data shape: {prices_df.shape}")
            print("sell_prices.csv boyut bilgileri:", prices_df.shape)
        except FileNotFoundError:
            #raise FileNotFoundError("sell_prices.csv file not found. Please check the file path.")
            raise FileNotFoundError("sell_prices.csv dosyasi bulunamadi. Lütfen dosya yolunu kontrol edin.")
    except FileNotFoundError:
        #raise FileNotFoundError("One or more dataset files not found. Please check the file paths.")
        #print("Please, use create_sample_datasets.py to create sample datasets.")
        # raise FileNotFoundError("Bir veya daha fazla veriset dosyasi bulunamadi. Lütfen dosya yollarini kontrol edin.")
        print("Lütfen, ornek verisetleri olusturmak icin create_sample_datasets.py dosyasini kullanin.")
        return None, None, None
            
        
    # filter CA_1 store and foods category (CA_1 magazasi ve yiyecek kategorisini filtrele)
    #print("Filtering data for store CA_1 and category FOOD...")
    print("CA_1 magazasi ve FOOD kategorisi icin veriler filtreleniyor...")
    ca1_mask = (sales_train_val_df['store_id'] == 'CA_1')
    ca1_sales = sales_train_val_df[ca1_mask].copy()
    
    #print("Number of products in CA_1 store:", ca1_sales.shape[0])
    print(f"CA_1 magazasindaki urun sayisi: {ca1_sales.shape[0]}")
    
    foods_mask = (ca1_sales['category_id'] == 'FOODS')
    food_sales = ca1_sales[foods_mask].copy()
    #print(f"Number of FOOD products in CA_1 store: {food_sales.shape[0]}")
    print(f"CA_1 magazasindaki FOOD kategorisi urun sayisi: {food_sales.shape[0]}")
    
    
    if len(food_sales) == 0:
        #print("No FOOD category products found in CA_1 store. We will proceed with all products from CA_1 store.")
        print("CA_1 magazasinda FOOD kategorisi urunu bulunamadi. CA_1 magazasindaki tum urunlerle devam edilecek.")
        food_sales = ca1_sales.copy()
    
    # top 5 sold items (en cok satan 5 urun)
    #print("Selecting top 5 sold items...")
    print("En cok satan 5 urun seciliyor...")
    sales_cols = [col for col in food_sales.columns if col.startswith('d_')]
    #print(f"Total day columns found: {len(sales_cols)}")
    print(f"Toplam gun kolon sayisi: {len(sales_cols)}")
    
    food_sales['total_sales'] = food_sales[sales_cols].sum(axis=1)
    
    top5_food_items = food_sales.nlargest(5, 'total_sales')
    # print("Top 5 sold items selected:")
    print("En cok satan 5 urun secildi:")
    for idx, (_, item) in enumerate(top5_food_items.iterrows(), start=1):
        print(f"{idx}. Item ID: {item['item_id']}, Toplam Satis: {item['total_sales']}")
        
        
    # daily sales time series 
    print(f"Gunluk satis zaman serisi olusturuluyor...")
    selected_items = top5_food_items[['id','item_id', 'store_id'] + sales_cols].copy()
    
    
    # formatini uzuna cevir (convert format to long)
    long_data = []
    
    
    for _, item_row in selected_items.iterrows():
        item_id = item_row['item_id']
        store_id = item_row['store_id']
        
        
        # her gun icin satis verisi
        for day_col in sales_cols:
            sales_value = item_row[day_col]
            
            if pd.isna(sales_value):
                sales_value = 0
                
            long_data.append({
                'item_id':item_id
                ,'store_id':store_id
                ,'d':day_col
                ,'sales':int(sales_value)
            })
    # create dataframe from long data (uzun veriden dataframe olustur)
    sales_long_df = pd.DataFrame(long_data)
    
    # merge with calendar to get dates (tarihi almak icin calendar ile birlestir)
    sales_long_df = sales_long_df.merge(calendar_df[['d','date']], on = 'd', how = 'left')
    
    # sort values (degerleri sirala)
    sales_long_df = sales_long_df.sort_values(['item_id','date']).reset_index(drop=True)
    
    
    print(f"Uzun Formatli Verinin Boyutu: {sales_long_df.shape}")
    print(f"Tarih Araligi: {sales_long_df['date'].min()} - {sales_long_df['date'].max()}")
    print(f"Toplam Gun Sayisi: {sales_long_df['date'].nunique()}")
    
    
    # impute missing dates with 0 sales (eksik tarihleri 0 satis ile doldur)
    print("Eksik Gunler Kontrol Ediliyor...")
    all_dates = pd.date_range(start=sales_long_df['date'].min(), end=sales_long_df['date'].max(), freq='D')
    
    complete_data = []
    
    for item_id in sales_long_df['item_id'].unique():
        item_data = sales_long_df[sales_long_df['item_id'] == item_id].copy()
        store_id = item_data['store_id'].iloc[0]
        
        
        # find missing dates (eksik tarihleri bul)
        existing_dates = set(item_data['date'])
        missing_dates = [date for date in all_dates if date not in existing_dates]
        
        if missing_dates:
            print(f"ID numarasi {item_id} icin {len(missing_dates)} eksik gun bulundu. Eksik gunler 0 satis ile dolduruluyor...")
            
            for missing_date in missing_dates:
                complete_data.append({
                    'item_id': item_id
                    ,'store_id': store_id
                    ,'date': missing_date
                    ,'sales':0
                })
            
            # add existing data (mevcut veriyi ekle)
        for _ , row in item_data.iterrows():
            complete_data.append({
                'item_id': row['item_id']
                ,'store_id': row['store_id']
                ,'date': row['date']
                ,'sales': row['sales']
            })
    
    # Tam veri setini dataframe e cevir (convert complete dataset to dataframe)
    complete_sales_df = pd.DataFrame(complete_data)
    complete_sales_df = complete_sales_df.sort_values(['item_id','date']).reset_index(drop=True)
    print(f"Tam Veri Seti Boyutu: {complete_sales_df.shape}")
    
    
    # sort values (degerleri sirala)
    all_dates_sorted = sorted(complete_sales_df['date'].unique())
    
    # split validation and train sets (dogrulama ve egitim setlerine bol)
    validation_days = 28
    
    if len(all_dates_sorted) <= validation_days:
        print(f"Yeterli gun sayisi yok: {len(all_dates_sorted)}. Dogrulama gun sayisi: {validation_days}")
        validation_days = max(1, len(all_dates_sorted) // 4) #verinin %25'ini ayir
        print(f"Verinin %25'i dogrulama icin ayrildi. Yeni dogrulama gun sayisi: {validation_days}")
    
    # tarih bolme tarihi belirle (determine date to split)
    split_date = all_dates_sorted[-validation_days]
    train_end_date = all_dates_sorted[-(validation_days + 1)] if len(all_dates_sorted) > validation_days else all_dates_sorted[0]
    
    
    # train and validation sets (egitim ve dogrulama setleri)
    train_df = complete_sales_df[complete_sales_df['date'] <= train_end_date].copy()
    valid_df = complete_sales_df[complete_sales_df['date'] >= split_date].copy()
    
    print(f"Egitim Seti Icin: {train_df['date'].min()} - {train_df['date'].max()}, Boyut: {train_df.shape}")
    print(f"Dogrulama Seti Icin: {valid_df['date'].min()} - {valid_df['date'].max()}, Boyut: {valid_df.shape}")
    
    
    # tarihi index olarak ayarla (set date as index)
    train_df = train_df.set_index('date')
    valid_df = valid_df.set_index('date')
    
    print(f"Sonular Kaydediliyor...")
    
    # CSV files 
    
    train_path = './artifacts/datasets/train.csv'
    validation_path = './artifacts/datasets/validation.csv'
    
    train_df.to_csv(train_path)
    valid_df.to_csv(validation_path)
    
    print(f"Egitim Verisi Kaydedildi: {train_path}")
    print(f"Dogrulama Verisi Kaydedildi: {validation_path}")
    
    # visualizations (gorsellestirme)
    print(f"Gun Bazinda Toplam Satis Grafikleri Olusturuluyor...")
    daily_total = complete_sales_df.groupby('date')['sales'].sum().reset_index()
    
    plt.figure(figsize = (12, 8))
    
    train_dates = train_df.reset_index()['date'].unique()
    valid_dates = valid_df.reset_index()['date'].unique()
    
    
    train_total = daily_total[daily_total['date'].isin(train_dates)]
    valid_total = daily_total[daily_total['date'].isin(valid_dates)]
    
    # egitim verisi (training data)
    plt.plot(train_total['date'], train_total['sales'], label = 'Egitim Verisi', color = 'forestgreen', linewidth=1.5)
    
    plt.plot(valid_total['date'], valid_total['sales'], label = 'Dogrulama Verisi', color = 'orange', linewidth=1.5)
    
    # split cizgisi (split line)
    plt.axvline(x = split_date
                ,color = 'red'
                ,linestyle='--'
                ,alpha = 0.6
                ,label = f'Dogrulama Baslangici: ({split_date.strftime("%Y-%m-%d")})'
)
    
    plt.title('Secilen 5 Urun Icin Gunluk Toplam Satis\n' + f'CA_1 Magazasi FOODS Kategorisi'
              ,fontsize=14
              ,fontweight='bold'
              ,color = 'midnightblue')
    plt.xlabel('Tarih', fontsize=12, fontweight='bold', color = 'midnightblue')
    plt.ylabel('Satis Miktari', fontsize=12, fontweight='bold', color = 'midnightblue')
    plt.legend(fontsize = 12)
    plt.grid(True, alpha = 0.4)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    
    # save figure (sekli kaydet)
    figure_path = './artifacts/figures/daily_total_sales.png'
    plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
    print("Grafik Kaydedildi:", figure_path)
    plt.close()
    
    
    print(f"Hizli Bir Bakis Icin Ozet Bilgiler Getiriliyor...")
    print(f"Secilen Urunler: {','.join(complete_sales_df['item_id'].unique())}")
    print(f"Toplam Gun Sayisi: {len(all_dates_sorted)}")
    print(f"Egitim Gun Sayisi: {len(train_df.reset_index()['date'].unique())}")
    print(f"Dogrulama Gun Sayisi: {len(valid_df.reset_index()['date'].unique())}")
    print(f"Maksimum Gunlik Satis: {complete_sales_df['sales'].max()}")
    print(f"Minimum Gunlik Satis: {complete_sales_df['sales'].min()}")
    
    
    print("Urun Bazinda Onemli Istatistikler:")
    items_stats= complete_sales_df.groupby('item_id')['sales'].agg(['min','max','mean','median','std']).round(2)
    for item_id , stats in items_stats.iterrows():
        print(f"Urun ID: {item_id} => Min: {stats['min']:.2f}, Max: {stats['max']:.2f}, Mean: {stats['mean']:.2f}, Median: {stats['median']:.2f}, Std: {stats['std']:.2f}")
        
    print("Veri Hazirlama Tamamlandi.")
    return train_df, valid_df, daily_total

def main():
    result = create_m5_subset()
    if result is None or (isinstance(result, tuple) and result[0] is None):
        print("Veri hazirlama basarisiz oldu.")
        return False
    else:
        print(f"CA_1 Magazasi FOODS Kategorisi Icin Veri Hazirlama Basarili.")
        return True
    
if __name__ == '__main__':
    try:
        result = create_m5_subset()
        if result is None or (isinstance(result, tuple) and result[0] is None):
            print(f"\nVeri dosyası bulunamadi.")
        else:
            train_data, valid_data, daily_sales = result
            print(f"M5 Veri Setinin Alt Kumesi Basariyla Olusturuldu.")
        
    except Exception as err:
        print(f"\nBeklenmeyen Bir Hata ile Karsilasildi: {err}")
        import traceback
        print("Sistemden Cikiliyor...")
        traceback.print_exc() 
        
    