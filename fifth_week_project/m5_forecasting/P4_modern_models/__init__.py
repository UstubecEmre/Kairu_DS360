""" 
P4: Modern Models Modulu
Prophet modeli kullanilarak tek bir urun icin tahminde bulunulur.

Icerisinde Bulunanlar:
    - prophet_single_item.py
"""

version = "1.0.0"

try:
    from .prophet_single_item import main as run_prophet
    __all__ = ['run_prophet']
except ImportError as err:
    print(f"P4: Modern Models Modulunde Hata Olustu: {err}")
    __all__ = []
