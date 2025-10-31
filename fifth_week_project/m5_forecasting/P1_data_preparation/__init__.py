"""

P1 : Data Preparation
Veri setinin hazir hale getirilmesini saglayan moduldur

Icerisinde Yer Alan Moduller:
    - create_m5_subset.py
    - create_sample_data.py

"""
__version__ = "1.0.0"


try:
    from .create_m5_subset import main as create_subset
    from .create_sample_data import main as create_sample
    
    __all__ = ['create_subset', 'create_sample']
except ImportError as err:
    print(f"P1 Data Preparation Modulu Yuklenemedi: {err}")
    __all__ = [] # bos liste don    