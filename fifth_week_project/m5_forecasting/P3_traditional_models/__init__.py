""" 
P3: Traditional Models Modulu
Bu modelde ARIMA yontemi kullanilmaktadir.

Icerisinde Bulunanlar:
    - arima_single_item.py

"""
version = "1.0.0"

try:
    from .arima_single_item import main as run_arima
    __all__ = ['run_arima']
except ImportError as err:
    print(f"P3: Traditional Models Modulunda Hata Olustu: {err}")
    __all__ = []