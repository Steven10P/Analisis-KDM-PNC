# ==========================================
# PARCHE MLOps: Hacer tolerante el __init__.py
# ==========================================
codigo_init = """# Archivo: src/models/__init__.py

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
"""

with open("/content/Analisis-KDM-PNC/src/models/__init__.py", "w") as f:
    f.write(codigo_init)

print("[✅] Archivo src/models/__init__.py actualizado con tolerancia a fallos.")
