# Archivo: src/models/pnc_factory.py

import torch
import torch.nn as nn
import torch.optim as optim
from circuits.pncrc import GenDisPNCRC

def build_pnc_model(config, device):
    """
    Construye e inicializa un Probabilistic Neural Circuit (GenDisPNCRC).
    """
    # 1. Instanciación del modelo enviándolo a GPU/CPU
    modelo_pnc = GenDisPNCRC(
        height=config.get('height', 28), 
        width=config.get('width', 28), 
        components=config['components'],
        n_classes=config.get('n_classes', 10), 
        mixing=config.get('mixing', 'sum')
    ).to(device)

    # 2. Configuración del optimizador (PNC en el paper usa SGD con momentum)
    optimizer = optim.SGD(
        modelo_pnc.parameters(), 
        lr=config['lr'], 
        momentum=config['momentum']
    )
    
    # 3. Función de pérdida estándar para clasificación en PyTorch
    criterion = nn.CrossEntropyLoss()
    
    # 4. Conteo de parámetros entrenables para comparar equitativamente con KDM
    total_params = sum(p.numel() for p in modelo_pnc.parameters() if p.requires_grad)

    return modelo_pnc, optimizer, criterion, total_params
