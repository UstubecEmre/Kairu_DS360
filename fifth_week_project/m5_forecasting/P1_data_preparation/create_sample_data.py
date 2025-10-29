#%% Gerekli kutuphanelerin icerye aktarilmasi (import required libraries)
import os
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import timedelta, datetime


def create_sample_m5_data():
    print("Ornek M5 veri seti olusturuluyor...") 
    
    # Tarih araligi belirleme
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(date_range)
    
    print(f"Tarih Araligi: {start_date.date()} - {end_date.date()} ({n_days} gun)")
    
    print("Takvim Verisi Olusturuluyor...")
    
    calendar_data = []
    
    for idx, date in enumerate(date_range):
        calendar_data.append({
            'd':f'd_{idx+1}'
            ,'date':date.strftime('%Y-%m-%d')
            ,'wm_yr_wk': date.isocalendar()[1] # Yilin haftasi
            ,'weekday':date.weekday() + 1 # Haftanin gunu 0 Pazartesi 6 pazar
            ,'month':date.month
            , 'year':date.year
            
            # Tatil bilgileri (ornek olarak rastgele atandi)
            ,'event_name_1': ''  # Boş bırak
            ,'event_type_1': ''
            ,'event_name_2': ''
            ,'event_type_2': ''
            
            #Nadir olaylar icin snap degerleri
            ,'snap_CA': np.random.choice([0, 1], p=[0.9, 0.1]) 
            ,'snap_TX': np.random.choice([0, 1], p=[0.9, 0.1])
            ,'snap_WI': np.random.choice([0, 1], p=[0.9, 0.1])
            
        })
        
    calendar_df = pd.DataFrame(calendar_data)
    
    print("Satis Verileri Olusturuluyor...")
    
    # urunlerin kategorileri
    states = ['CA', 'TX', 'WI']
    stores_per_state = 4 # her eyalette 4 magaza
    item_per_store = 10 #10 adet urun
    
    sales_data = [] # satis verisi listesi
    item_counter = 0
    
    for state in states:
        # Yalnizca CA kullanilsin
        if state != 'CA':
            continue
        # sadece 1 magaza kullanilsin
        for store_num in range(1, 2):
            store_id = f'Store_{state}_{store_num}'
            
            for item_num in range(1, item_per_store + 1):
                item_counter += 1
                item_id= f'Item_{item_counter:03d}'
                
                # urunlerin kategorileri
                dept_id = f'Dept_{(item_counter % 3) + 1}' # 3 departman
                cat_id = f'Cat_{(item_counter % 2) + 1}' # 2 kategori
                
                # id bilgileri
                id_str = f'{item_id}_{dept_id}_{cat_id}_{store_id}_validation'
                
                # satis verileri olusturalim
                base_demand = np.random.uniform(10, 50) # temel talep
                trend = np.linspace(0, 5, n_days) # zamanla artis
                
                
                # haftalik mevsimsellik (hafta ici ve hafta sonu farki)
                weekly_pattern = []
                for day in date_range:
                    if day.weekday() < 5: # hafta ici
                        weekly_pattern.append(1.0)
                    else: # hafta sonu
                        weekly_pattern.append(1.3)
                weekly_pattern = np.array(weekly_pattern)
                
                # Aylik mevsimsellik
                monthly_pattern = []
                for date in date_range:
                    if date.month in [11, 12]: # kasim, aralik
                        monthly_pattern.append(1.5)
                    elif date.month in [6, 7, 8]: # yaz aylarinda biraz dusuk talep
                        monthly_pattern.append(1.2)
                    else:
                        monthly_pattern.append(1.0)
                monthly_pattern = np.array(monthly_pattern)
                
                
                # Gercek veriye benzemesi icin gurultu ekleyelim
                noise = np.random.normal(0, base_demand * 0.1, n_days)
                
                # final talep hesaplama
                sales_values = base_demand + trend + \
                              (base_demand * (weekly_pattern - 1)) + \
                              (base_demand * (monthly_pattern - 1)) + \
                              noise
                
                # negatif satislari sifira cevir
                sales_values = np.where(sales_values < 0, 0, sales_values)
                
                # hic satis yapilmayan gunler icin 0 degeri
                zero_mask = np.random.random(n_days) < 0.05 # %5 ihtimalle 0 satis
                sales_values[zero_mask] = 0
                
                sales_values = sales_values.round().astype(int)
                
                
                # satir verisi olustur
                row = {
                    'id': id_str,
                    'item_id': item_id,
                    'dept_id': dept_id,
                    'cat_id': cat_id,
                    'store_id': store_id,
                    'state_id': state,
                }
                
                # satis degerlerini ekleyelim
                for idx, value in enumerate(sales_values):
                    row[f'd_{idx+1}'] = value
                
                sales_data.append(row)
    sales_df = pd.DataFrame(sales_data)
    print(f"{len(sales_df)} Urun Icin {n_days} Gunluk Satis Verisi Olusturuldu")
    
    
    # Satis verisi olusturalim
    print("Fiyat Verileri Olusturuluyor...")
    prices_data =[]
    for _, row in sales_df.iterrows():
        base_price = np.random.uniform(5, 50) # temel fiyat
        
        weeks = calendar_df['wm_yr_wk'].unique()
        for week in weeks[:20]:# ilk 20 hafta icin fiyat verisi
            price_variation = np.random.uniform(0.9, 1.1) # fiyat degisimi
            final_price = round(base_price * price_variation, 2)
            
            prices_data.append({
                'store_id': row['store_id']
                ,'item_id': row['item_id']
                ,'wm_yr_wk': week
                ,'sell_price': final_price
            })
    # fiyat verisi dataframe        
    prices_df = pd.DataFrame(prices_data)
    
    # dosyalari kaydetme
    def ensure_dir(dir_path):
        p = Path(dir_path) 
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
        else:
            print("Dizin zaten mevcut:", p)
    
    ensure_dir('./data')
    data_dir = './data'
    calendar_path = os.path.join(data_dir,'calendar.csv')
    sales_path = os.path.join(data_dir,'sales_train_validation.csv')
    prices_path = os.path.join(data_dir,'sell_prices.csv')
    
    calendar_df.to_csv(calendar_path, index=False)
    sales_df.to_csv(sales_path, index=False)
    prices_df.to_csv(prices_path, index=False)
    
    print("Ornek Veriler Olusturuldu")
    print(f"{calendar_path} Dosya Yolundaki Takvim Verisi Boyutu: {calendar_df.shape}")
    print(f'{sales_path} Dosya Yolundaki Satis Verisi Boyutu: {sales_df.shape}')
    print(f'{prices_path} Dosya Yolundaki Fiyat Verisi Boyutu: {prices_df.shape}')
    
    # Veri ozeti gosterelim
    print("Veri Istatistikleri:")
    print(f"Toplam Gun Sayisi: {n_days}")
    print(f"Toplam Urun Sayisi: {len(sales_df)}")
    print(f"Ortalama Gunluk Satis: {sales_df[[col for col in sales_df.columns if col.startswith('d_')]].mean().mean():.1f}")
    print(f"Maksimum Gunluk Satis: {sales_df[[col for col in sales_df.columns if col.startswith('d_')]].max().max()}")
    
    return calendar_df, sales_df, prices_df


if __name__ == "__main__":
    print("M5 Ornek Veri Seti Olusturuluyor...")
    try:
        create_sample_m5_data()
        print("M5 Ornek Veri Seti Basariyla Olusturuldu.")
    except Exception as err:
        print(f"Beklenmeyen Bir Hata Olustu: {err}")
        import traceback
        print("Uygulamadan Cikis Yapiliyor...")
        traceback.print_exc()