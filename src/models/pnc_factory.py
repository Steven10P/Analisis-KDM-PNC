import torch
import torch.nn as nn
import torch.optim as optim
from circuits.pncrc import GenDisPNCRC

def build_pnc_model(config, device):
    """
    Construye e inicializa un Probabilistic Neural Circuit (GenDisPNCRC) de forma dinámica.
    """
    # 1. Instanciación del modelo enviándolo a GPU/CPU
    # Todo es dictado por el YAML, con valores por defecto de seguridad
    modelo_pnc = GenDisPNCRC(
        height=config.get('height', 28), 
        width=config.get('width', 28), 
        components=config['components'],
        n_classes=config.get('n_classes', 10), 
        mixing=config.get('mixing', 'sum')
    ).to(device)

    # 2. Configuración del optimizador dinámico
    opt_name = config.get('optimizer', 'sgd').lower()
    
    if opt_name == 'adam':
        optimizer = optim.Adam(
            modelo_pnc.parameters(), 
            lr=config['lr']
            # Adam maneja sus propios momentos internamente
        )
    elif opt_name == 'sgd':
        optimizer = optim.SGD(
            modelo_pnc.parameters(), 
            lr=config['lr'], 
            momentum=config.get('momentum', 0.9) # Requiere momentum
        )
    else:
        raise ValueError(f"Optimizador '{opt_name}' no soportado. Usa 'sgd' o 'adam'.")
    
    # 3. Función de pérdida estándar para clasificación
    criterion = nn.CrossEntropyLoss()
    
    # 4. Conteo de parámetros entrenables para comparar equitativamente con KDM
    total_params = sum(p.numel() for p in modelo_pnc.parameters() if p.requires_grad)

    return modelo_pnc, optimizer, criterion, total_params
