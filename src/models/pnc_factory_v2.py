import torch
import torch.nn as nn
import torch.optim as optim
from circuits.pncrc import GenDisPNCRC



import torch
import torch.nn as nn
import torch.optim as optim
from circuits.pncrc import GenDisPNCRC # Asegúrate de importar de tu ruta correcta

def build_pnc_model(config, device):
    """Construye e inicializa un PNC nativo de PyTorch para SVHN (32x32)."""
    
    modelo_pnc = GenDisPNCRC(
        height=config.get('height', 32), # SVHN es 32x32
        width=config.get('width', 32),   # SVHN es 32x32
        components=config['components'],
        n_classes=config.get('n_classes', 10), 
        mixing=config.get('mixing', 'sum')
    ).to(device)

    opt_name = config.get('optimizer', 'adam').lower()
    
    if opt_name == 'adam':
        optimizer = optim.Adam(modelo_pnc.parameters(), lr=config['lr'])
    elif opt_name == 'sgd':
        optimizer = optim.SGD(modelo_pnc.parameters(), lr=config['lr'], momentum=config.get('momentum', 0.9))
    
    criterion = nn.CrossEntropyLoss()
    total_params = sum(p.numel() for p in modelo_pnc.parameters() if p.requires_grad)

    return modelo_pnc, optimizer, criterion, total_params

def build_pnc_model_v2(config, device):
    """
    Construye e inicializa un Probabilistic Neural Circuit nativo de PyTorch.
    """
    # Instanciación con soporte explícito para canales (SVHN = 3)
    modelo_pnc = GenDisPNCRC(
        height=config.get('height', 32), 
        width=config.get('width', 32), 
        channels=config.get('channels', 3), # CRÍTICO: Soporte multicanal
        components=config['components'],
        n_classes=config.get('n_classes', 10), 
        mixing=config.get('mixing', 'sum')
    ).to(device)

    opt_name = config.get('optimizer', 'adam').lower()
    
    if opt_name == 'adam':
        optimizer = optim.Adam(modelo_pnc.parameters(), lr=config['lr'])
    elif opt_name == 'sgd':
        optimizer = optim.SGD(modelo_pnc.parameters(), lr=config['lr'], momentum=config.get('momentum', 0.9))
    else:
        raise ValueError(f"Optimizador '{opt_name}' no soportado.")
    
    criterion = nn.CrossEntropyLoss()
    total_params = sum(p.numel() for p in modelo_pnc.parameters() if p.requires_grad)

    return modelo_pnc, optimizer, criterion, total_params
