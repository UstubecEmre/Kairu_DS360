## **1. Loan Credit Veri Seti:**

Loan Credit veri seti ile çalıştık.
Bu veri setinde, bir kişinin kredisini ödeyip ödemeyeceğini problemini çözmeye yönelik
makine öğrenmesi modellerini kullanarak bir model geliştirmek hedeflenmektedir.

Bu veri setiyle çalışmak isterseniz:
**[Loan Data Veri Seti](https://www.kaggle.com/datasets/zhijinzhai/loandata)**

Bu veri setinde, sınıflandırma problemi üzerinde makine öğrenmesi algoritmalarından
Random Forest Classifier, Logistic Regression ve XGBoost modelleri ve bunların undersampled, smote vb. yöntemlerle dengeli hale getirilmeye çalışılan halleri ile çalışılmıştır.

Veri seti küçük olduğundan kaynaklı, aşırı öğrenmeye eğilim göstermektedir. Bunun önüne geçilebilmesi amacıyla, güçlü makine öğrenmesi modellerini kullanmamak veya cross-validation (çapraz doğrulama) gibi validasyon yöntemlerinin kullanılması önerilir.

## **2. Projenin Klonlanması ve Kurulumların Gerçekleştirilmesi**

```bash
git clone git@github.com:UstubecEmre/Kairu_DS360.git

pip install -r requirements.txt
```

## **3. Proje Yapısı**:

```text
third_week_project/
└── loan_data_analysis/
    ├── dvc/
    │   ├── cache/
    │   ├── tmp/
    │   ├── .gitignore
    │   └── config
    │
    ├── artifacts/
    │   ├── .gitignore
    │   ├── feature_schema_cw.json
    │   ├── feature_schema_smote.json
    │   ├── feature_schema_under.json
    │   ├── model_logisticregression_cw.pkl
    │   ├── model_logisticregression_smote.pkl
    │   ├── model_logisticregression_under.pkl
    │   ├── model_randomforest_cw.pkl
    │   ├── model_randomforest_smote.pkl
    │   ├── model_randomforest_under.pkl
    │   ├── model_xgb_cw.pkl
    │   ├── model_xgb_smote.pkl
    │   ├── model_xgb_under.pkl
    │   └── preprocessor_cw.pkl
    │
    ├── data/
    │   ├── processed/
    │   │   ├── class_weights.json
    │   │   ├── test_original.csv
    │   │   ├── train_original.csv
    │   │   ├── train_smote.csv
    │   │   └── train_undersampled.csv
    │   │
    │   └── raw/
    │       ├── loan_data.csv
    │       └── .gitignore
    │
    ├── notebooks/
    │   ├── check_version.ipynb
    │   └── loan_data_eda.ipynb
    │
    ├── src/
    │   ├── __pycache__/
    │   ├── data_loader.py
    │   ├── eda.py
    │   ├── preprocessing.py
    │   ├── train_with_mlflow.py
    │   └── train.py
    │
    ├── streamlit_app/
    │   └── app.py
    │
    ├── .dvcignore
    ├── .gitignore
    ├── dvc.lock
    ├── dvc.yaml
    ├── README.md
    └── requirements.txt
```

## **4. Kullanılan Teknolojiler:**

Bu veri seti ile çalışılırken:

- DVC (Data Version Control)
- Streamlit
- MLflow
  kullanılmıştır.

## **5. Kodların Çalıştırılması**

dvc.yaml dosyasını çalıştırmak isterseniz, öncelikle ilgili klasörünüze **dvc init** dahil etmeniz gerekmektedir. Ardından dvc.yaml dosyanızı oluşturduktan sonra yapmanız gereken:

```bash
dvc repro
```

Eğer ki mlflow üzerinden modelinizin metriklerinin nasıl değiştiğini görmek isterseniz yapmanız gereken:

```bash
mlflow ui
```

## **Teşekkür**

Bu projeyi, **KAIRU** eğitim platformunun çok değerli eğitimi olan **DS360** eğitimi kapsamında, saygı değer eğitmenimiz **Yasemin Arslan** ve değerli mentörlerimizin destekleri sayesinde gerçekleştiriyoruz. Bu nedenle, bizlere yardımlarını eksik etmeyen eğitmenimize ve mentörlerimize teşekkür ediyorum.
