# Archivo: src/models/__init__.py

# Intentamos cargar KDM (Funcionará en el notebook 02a)
try:
    from .kdm_factory import build_kdm_model
except ImportError:
    pass

# Intentamos cargar PNC (Funcionará en el notebook 02b)
try:
    from .pnc_factory import build_pnc_model
except ImportError:
    pass
