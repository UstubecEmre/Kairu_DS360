# **IRIS Veri Seti MLOPS Projesi**

Bu çalışmada, **Iris veri seti** kullanılarak MLOps süreçlerinin nasıl gerçekleştirilebileceği gösterilmektedir.

## **Projenin Amacı**

Projenin amacı, **çoklu sınıflandırma** problemine sahip olan Iris veri setini kullanarak makine öğrenmesi modelleri kullanılarak tahminlemenin yapılması ve bunun dağıtımının sağlanmasıdır.

## **Proje Yapısı**

Projenin kolay yönetilebilir olmasının sağlanması amacıyla proje yapısı şu şekildedir.

```text
iris-mlops/
├── src/                        # Tüm Python kaynak kodları (Git tarafından takip edilir)
│   ├── clean_data.py           # Veri temizleme ve ön işleme
│   ├── download_data.py        # Veri indirme/çekme
│   ├── train_model.py          # Model eğitimi ve değerlendirme
│   └── train_model_mlflow.py   # (Opsiyonel) MLflow ile özel olarak eğitim/izleme kodu
│
├── data/                       # Veri dosyaları (DVC tarafından takip edilir, Git tarafından görmezden gelinir)
│   ├── raw/                    # İlk indirilen, ham veri
│   └── processed/              # Temizlenmiş ve işlenmiş veri
│
├── models/                     # Eğitilmiş modeller ve metrikler (DVC tarafından takip edilir)
│   ├── random_forest_model.pkl # Eğitilmiş model dosyası (Örnek)
│   ├── features.json           # Modelin kullandığı son özellik listesi
│   └── metrics.json            # DVC tarafından izlenecek model performans metrikleri
│
├── .dvc/                       # DVC Yapılandırması (Git tarafından takip edilir)
│   ├── cache/                  # DVC Veri Önbelleği (Git/DVC tarafından görmezden gelinir)
│   └── config                  # DVC ayarları (remote/cache konumu)
│
├── dvc.yaml                    # Pipeline Tanımı: Aşamaları, bağımlılıkları ve çıktıları içerir.
├── dvc.lock                    # Veri ve model durumlarının hash'lerini tutar
├── .dvcignore                  # DVC'nin görmezden geleceği dosyalar
├── .gitignore                  # Git'in görmezden geleceği dosyalar (mlruns/, data/, *.pkl vb.)
├── requirements.txt            # Python bağımlılıkları listesi
└── README.md                   # Projenin ana açıklaması ve kullanım kılavuzu
```

## **Projenin Klonlanması ve Kurulumların Gerçekleştirilmesi**

```bash
git clone git@github.com:UstubecEmre/Kairu_DS360.git
cd iris-mlops
pip install -r requirements.txt
```

## **Kullanım**

```bash
dvc repro #dvc dosyalarını çalıştırmak için
mlflow ui # MLflow üzerinden takip edebilmek için
```

## **Beklenen Çıktılar**:

Bu dosyaları çalıştırdığınız takdirde **models/metrics.json** dosyamızın içerisinde
en iyi accuracy score ve f1 score değerlerini ve bu modellerin hangileri olduğunu göreceksiniz.

## **Kullanılan Teknolojiler**

- DVC (Data Version Control)
- MLFlow
- Scikit-learn

## **Teşekkür**

Bu projeyi, **KAIRU** eğitim platformunun çok değerli eğitimi olan **DS360** eğitimi kapsamında, saygı değer eğitmenimiz **Yasemin Arslan** ve değerli mentörlerimizin destekleri sayesinde gerçekleştiriyoruz. Bu nedenle, bizlere yardımlarını eksik etmeyen eğitmenimize ve mentörlerimize teşekkür ediyorum.
