#%% import required libraries (gerekli kutuphaneleri iceri aktar)
import os
import joblib
import streamlit as st 
import json 
from datetime import datetime
import pandas as pd 
import numpy as np

#%% set a title 
st.set_page_config(page_title = "Kredi Risk Analizi", layout= 'centered')
st.title('Kredi Risk Paneli - Canli Skor')


#%% set file paths (dosya yollarini ayarla)
MODEL_PATH = "artifacts/model_xgb_smote.pkl"
SCHEMA_PATH = "artifacts/feature_schema_smote.json"
PRE_PATH = "artifacts/preprocessor_smote.pkl"


#%% clean cache  (her seferinde tekrar yuklememesi icin on bellegi temizle)
@st.cache_resource
def load_artifacts():
    """ artifacts klasorundaki dosyalari yukler"""
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PRE_PATH)
        with open(SCHEMA_PATH, 'r') as file:
            schema_cols = json.load(file)['columns']
        return model, preprocessor, schema_cols
    except FileNotFoundError:
        raise FileNotFoundError(f"Hatali Bir Dosya Yolu Girdiniz")
    except Exception as err:
        raise Exception(f"Beklenmedik Bir Hata Olustu. Hata Nedeni: {err}")


model, preprocessor, schema_columns = load_artifacts()


#%% define to_timestamp function
def to_timestamp(d):
    return pd.to_datetime(d) // 10**9


def build_features_single(Principal, terms, age, education, Gender, eff, due):
    eff_timestamp = to_timestamp(eff)
    due_timestamp = to_timestamp(due)
    planned = (pd.to_datetime(due_timestamp) - pd.to_datetime(eff_timestamp)).days
    principal_per_term = (Principal / terms) if terms else np.nan
    
    
    df = pd.DataFrame(
        [
            {
                "Principal": Principal
                ,"terms": terms
                ,"age": age
                ,"education": education
                ,"Gender": Gender
                ,"effective_date": eff_timestamp
                ,"due_date": due_timestamp
                ,"planned_term_days": planned
                ,"principal_per_term": principal_per_term
            }
        ]
    )
    # impute not a number rows
    df = df.reindex(columns = schema_columns, fill_value= np.nan)
    return df


#%% add widgets (widget ekleyelim)
# add subheader (alt baslik ekleyelim)
st.subheader("Tekil Basvuru (Deger Degistikce Skor Aninda Degisir)")

columns_1, columns_2 = st.columns(2)
with columns_1:
    Principal = st.number_input(label = "Principal"
                                 ,min_value = 0
                                 ,max_value =  1000
                                 ,step = 50
                                 ,key = "principal")
    
    terms = st.selectbox(
        label = "Terms (gun)"
        ,options = [7, 15, 30]
        ,index = 2
        ,key = "terms"
    )
    
    age = st.number_input(
        label = "Age"
        ,min_value = 18
        ,max_value = 120
        ,value = 30
        ,key = "age"
    )
    
with columns_2:
    education = st.selectbox(
        label = "Education"
        ,options = ['High School or Below','college','Bechalor','Master or Above']
        ,key = 'education'
    )
    Gender = st.selectbox(
        label = "Gender"
        ,options = ['male','female']
        ,key = "gender"
    )
    effective_date = st.date_input(
        label = "effective_date"
        ,value= datetime.today()
        ,key = "eff"
    )
    due_date = st.date_input(
        label = "due_date"
        ,value = datetime.today()
        ,key = "due"
    )

if due_date < effective_date:
    st.warning("Due date, effective date'den once olamaz!!!")

# render
X = build_features_single(
    Principal = Principal
    ,terms= terms
    ,age = age
    ,education = education
    ,Gender = Gender
    ,eff = effective_date
    ,due = due_date
)

X_transformed = preprocessor.transform(X)
proba_transformed = float(model.predict_proba(X_transformed)[:, 1][0])
st.metric(f"PAIDOFF Olasiligi: {proba_transformed:.2%}")
st.caption("Onemli Not: Olasilik dustukce risk orani da fazladir. Ters oranti soz konusudur")
st.divider()



# show all results (butun skorlari goster)
st.subheader("Toplu Skor (CSV Formatinda)")
uploaded_file = st.file_uploader(
    label = "CSV Yukleyin"
    ,type = ['csv'])

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    
    # convert from date to  timestamp (timestamp'e donusturelim)
    for column in ['due_date','effective_date']:
        if column in input_data.columns and not np.issubdtype(input_data[column].dtype, np.number):
            input_data[column] = pd.to_datetime(input_data[column]
                                          ,errors= 'coerce').view("int64") // 10 ** 9
    
    if {'Principal', 'terms'}.issubset(input_data.columns):
        input_data['principal_per_term'] = np.where(
            input_data['terms'] == 0
            , np.nan
            , input_data['terms']
            )
        
    
    if set(['effective_date', 'due_date']).issubset(input_data.columns):
        pass 
    
    X_bulk = input_data.reindex(columns = schema_columns, fill_value= np.nan)
    X_transformed_bulk = preprocessor.transform(X_bulk)
    probs = model.predict_proba(X_transformed_bulk)[:, 1]

    
    output_data = input_data.copy()
    output_data['paid_prob'] = probs
    st.dataframe(output_data.head())
    st.download_button("Sonuclari Indir (CSV Formatinda)"
                       ,output_data.to_csv(index = False).encode('utf-8')
                       ,file_name=f'scored_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
)