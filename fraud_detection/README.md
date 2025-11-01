### **1. Projenin Amacı:**

Dengesiz bir veri setine sahip olan verisetimiz [credit card fraud detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) de veri dengesizliğini gidererek bir sınıflandırma problemini çözmeye çalıştık.

#### 1.1. Veri Seti Tanıtımı:

Veri setimiz, bankalarda uygulanan regulasyonlara uygun olacak şekilde kişisel bilgileri içermeyecek bir şekilde maskelenmiş 28 adet sütunu içermektedir. (V1-V28) Time, Class, Amount sütunlarına da sahip olan veri setimiz toplamda 31 adet sütundan oluşmaktadır. İkili sınıflandırma problemleri için uygundur.

### **2. Projenin Yapısı:**

```bash
fourth_week_project/
└── fraud_detection/
    ├── config/
    │   └── configuration.yaml
    ├── data/
    │   ├── raw/
    │   │   └── creditcard_fraud.csv
    │   └── processed/
    │       ├── anomaly_scores_raw.csv
    │       ├── dataset_processed_supervised.csv
    │       ├── outlier_meta_raw.json
    │       ├── test_processed_supervised.csv
    │       └── train_processed_supervised.csv
    ├── logs/
    │   └── fraud_detection.log
    ├── mlartifacts/
    │   └── models/
    ├── notebooks/
    │   └── fraud_detection_eda.ipynb
    ├── src/
    │   ├── download_data.py
    │   ├── model_preprocessing.py
    │   ├── outlier_detection.py
    │   ├── model_evaluation.py
    │   ├── model_explainability_fraud.py
    │   └── fraud_detection_pipeline.py
    ├── .gitignore
    ├── mlflow.db
    ├── README.md
    └── requirements.txt
```

### **3. Sanal Ortamın Kurulması ve Projenin Çalıştırılması:**

Windows işletim sistemine sahip olanlar için

#### **3.1. Projeyi Klonlanmak:**

```bash
# Projeyi klonlamak için
git clone https://github.com/UstubecEmre/fourth_week_project.git
cd fourth_week_project/fraud_detection
```

#### **3.2. Sanal Ortamın Oluşturma ve Aktif Etme:**

```bash
# Sanal ortam oluşturma ve aktive etme (Windows)

python -m venv fourth_week_env
fourth_week_env\Scripts\activate

# Gerekli paketleri yükleme

pip install -r requirements.txt

```

### **4. Kullanılan Python Dosyaları:**

#### **4.1. Python Dosyalarını Çalıştırmak:**

Burada fraud_detection_pipeline.py dosyasını tek başına çalıştırarak bütün işlemleri gerçekleştirebileceğiniz gibi, sırasıyla şu şekildedir:

```python
python src/download_data.py
```

```python
python  src/preprocessing.py
```

```python
python src/outlier_detection.py
```

```python
python src/model_evaluation.py
```

```
python src/model_explainability_fraud.py
```

#### **4.2. Python Dosyalarının Tanıtımı:**

- download_data.py: Ham veriyi yükler.

- model_preprocessing.py: Eksik değerlerin doldurulması, normalizasyon/standardizasyon.

- outlier_detection.py: Aykırı değer tespiti (Isolation Forest, LOF vb.).

- model_evaluation.py: Modellerin başarı metriklerinin hesaplanması.

- model_explainability_fraud.py: Model açıklanabilirliği (SHAP ağırlıklı).

- fraud_detection_pipeline.py: Tüm adımları tek seferde çalıştırır.

```python
cd ilgili_klasor (burada fourth_week_project altindaki fraud_detection)

python src/fraud_detection_pipeline.py
```

### **5. Geliştirici**

[UstubecEmre](https://github.com/UstubecEmre)

#### **6. Teşekkürler:**

Bu eğitimde bizlere yol gösterici olan **Kairu** platformunun kurucusu ve değerli eğitmenimiz **Yasemin Arslan'a** ve her daim yardımını esirgemeyen **Kairu mentoruma** teşekkürlerimi sunarım.

[Daha fazla bilgi için veri seti](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

**Not**: Projeyi çalıştırırken GPU kullanmanız tavsiye edilir, CPU ile işlemler saatleri aşabilmektedir.
Bu anlamda benim gibi uzun süre bekleyip terminali sonlandırmak isterseniz **CTRL+C** kullanabilirsiniz.
