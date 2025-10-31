"""
P2: Feature Engineering Modulu
Oznitelik muhendisligini gerceklestiren moduldur.

Icerisinde Bulunanlar:
    - feature_engineering.py 
"""

version = "1.0.0"
try:
    from .feature_engineering import main as create_features
    
    __all__ = ['create_features']
except ImportError as err:
    print(f"P2 Feature Engineering Modulunde Hata Olustu: {err}")
    __all__ = []
    