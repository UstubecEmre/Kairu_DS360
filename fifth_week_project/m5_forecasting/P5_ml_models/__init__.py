""" 
P5: ML Models Modulu
Bu modul, Makine Ogrenmesinin guclu algoritmalarindan olan LightGBM kullanmaktadir.
ARIMA ve Prophet modellerinden farkli olarak coklu urun tahmini gerceklestirmektedir.

Icerisinde Bulunanlar:
    - lightgbm_multi_items.py

"""
__version__ = "1.0.0"

try:
    from .lightgbm_multi_items import main as run_lightgbm
    __all__ = ['run_lightgbm']
except ImportError as err:
    print(f"P5: ML Models Modulunde Hata Olustu: {err}")
    __all__ = []