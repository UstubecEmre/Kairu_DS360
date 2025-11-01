### **1. Projenin Amacı:**

Projemizde, dünya perakende şirketlerinden olan **Walmart'ın** satış verilerinin örneklemesinden ARIMA, Prophet, LightGBM gibi modelleri kullanarak satış tahminlerini gerçekleştirmek, modellerin birbiriyle olan karşılaştırmasını gözlemleyerek öğrenmek amaçlanmaktadır.

#### **1.1. Veri Seti**:

**M5 Veri seti**, 5 yıl önce gerçekleştirilen **University of Nicosia · Featured Prediction Competition** yarışmasında kullanılan bir veri setidir.

Zaman serisi analizleri için idealdir.

[M5 Veri Seti](https://www.kaggle.com/competitions/m5-forecasting-accuracy)

---

### **2. Proje Yapısı**

```text
fifth_week_project
│
└───m5_forecasting
    │
    ├───.gitignore
    ├───README.md
    ├───requirements.txt
    ├───run_modular.py
    │
    ├───notebooks
    │       models_info.ipynb
    │
    ├───artifacts
    │   ├───datasets
    │   │       fe_train.parquet
    │   │       fe_valid.parquet
    │   │       train.csv
    │   │       validation.csv
    │   │       X_train.parquet
    │   │       X_valid.parquet
    │   │       y_train.parquet
    │   │       y_valid.parquet
    │   │
    │   ├───figures
    │   │       correlations.png
    │   │       daily_total_sales.png
    │   │       feature_distributions.png
    │   │       lgbm_feature_importance.png
    │   │       prophet_FOODS_3_090_components.png
    │   │       prophet_FOODS_3_090_forecast.png
    │   │
    │   ├───models
    │   │       arima_FOODS_3_090.pkl
    │   │       lightgbm.pkl
    │   │       prophet_FOODS_3_090.json
    │   │
    │   └───predictions
    │           arima_forecast_FOODS_3_090.csv
    │           arima_report_FOODS_3_090.json
    │           arima_vs_prophet_FOODS_3_090.json
    │           lightgbm_forecast_all.csv
    │           lightgbm_report.json
    │           prophet_forecast_FOODS_3_090.csv
    │
    ├───P1_data_preparation
    │       __init__.py
    │       create_m5_subset.py
    │       create_sample_data.py
    │
    ├───P2_feature_engineering
    │       __init__.py
    │       feature_engineering.py
    │
    ├───P3_traditional_models
    │       __init__.py
    │       arima_single_item.py
    │
    ├───P4_modern_models
    │       __init__.py
    │       prophet_single_item.py
    │
    └───P5_ml_models
            __init__.py
            lightgbm_multi_items.py

```

### **3. Sanal Ortamın Kurulması ve Projelerin Çalıştırılması:**

#### **3.1. Projeyi Klonlamak:**

```bash
# Projeyi klonlamak için
git clone https://github.com/UstubecEmre/fifth_week_project.git
cd fifth_week_project/m5_forecasting
```

#### **3.2. Sanal Ortamın Oluşturma ve Aktif Etme:**

```bash
# Sanal ortam oluşturma ve aktive etme (Windows)

python -m venv fifth_week_env
fifth_week_env\Scripts\activate
# Gerekli paketleri yükleme
pip install -r requirements.txt

```

### **4. Kullanılan Python Dosyaları:**

- P1_data_preparation

  - create_m5_subset : # M5 Veri seti için alt küme oluşturur
  - create_sample_data.py # M5 Veri setini yüklemektedir.

- P2_feature_engineering

  - feature_engineering.py # Veri seti içerisinden LAG, ROLLING ve önemli DATE özellikleri üretir

- P3_traditional_models

  - arima_single_item.py # ARIMA modeli kullanılarak veri yüklenmesinden sonuçların kaydedilmesine kadar işlemleri gerçekleştirir

- P4_modern_models

  - prophet_single_item.py # Prophet modeli kullanılarak veri yüklenmesinden sonuçların kaydedilmesine kadar işlemleri gerçekleştirir

- P5_ml_models
  - lightgbm_multi_items.py # LightGBM modelİ kullanılarak veri yüklenmesinden sonuçların kaydedilmesine kadar işlemleri gerçekleştirir

### **5. Python Dosyalarının Modüler Çalıştırılması:**

```bash
# P1_data_preparation Modulu için
python run_modular.py --module P1

# P2_feature_engineering Modulü için
python run_modular.py --module P2

# P3_traditional_model Modülü için
python run_modular.py --module P3

# P4_modern_models Modülü için
python run_modular.py --module P4

# P5_ml_models Modülü için
python run_modular.py --module P5
```

> **Note:**
> Dikkat Edilmesi Gereken Kısımlar:

index_col = 'date' yapmayı sakın unutmayınız. Aksi takdirde veri tiplerinin uyumsuz olmasından kaynaklı işlemleriniz sırasında hata alabilirsiniz.

### **6. Geliştirici**

[UstubecEmre](https://github.com/UstubecEmre)

#### **7. Teşekkürler:**

Bu eğitimde bizlere yol gösterici olan **Kairu** platformunun kurucusu ve değerli eğitmenimiz **Yasemin Arslan'a** ve her daim yardımını esirgemeyen **Kairu mentoruma** teşekkürlerimi sunarım.
