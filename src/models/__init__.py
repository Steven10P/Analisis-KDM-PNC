# ==========================================
# EXPOSICIÓN DE FACTORÍAS (MLOps Pipeline)
# ==========================================

# 1. Factorías de Kernel Density Matrix (KDM)
try:
    # Versión original (Fashion-MNIST / Referencia inicial)
    from .kdm_factory import build_kdm_model
    # Versión evolucionada (SVHN / Imágenes Naturales)
    from .kdm_factory_v2 import build_kdm_model_v2
except ImportError:
    pass

# 2. Factorías de Probabilistic Neural Circuits (PNC)
try:
    # Mantendremos la misma lógica cuando implementemos PNC
    from .pnc_factory import build_pnc_model
    from .pnc_factory_v2 import  build_pnc_mode_v2
except ImportError:
    pass
