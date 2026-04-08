import torch
import torch.nn as nn
import torch.optim as optim

def build_pnc(config, device):
    """
    Fábrica para instanciar el Probabilistic Neural Circuit (PNC).
    """
    print(f"[*] Construyendo PNC (PyTorch) para {config['dataset'].upper()}...")
    
    # Extraer dimensiones espaciales y arquitectónicas
    h = config['arquitectura']['height']
    w = config['arquitectura']['width']
    n_classes = config['arquitectura']['n_classes']
    components = config['arquitectura'].get('components', 10)
    
    # Hiperparámetros de optimización
    lr = config['entrenamiento'].get('lr', 0.01)
    momentum = config['entrenamiento'].get('momentum', 0.9)

    # Instanciación del Modelo
    # Se intentará importar tu clase GenDisPNCRC
    try:
        from .pnc_core import GenDisPNCRC
        model = GenDisPNCRC(
            height=h, 
            width=w, 
            n_classes=n_classes, 
            num_components=components
        )
    except ImportError:
        print("[!] Advertencia: No se encontró 'pnc_core.py'. Usando un modelo dummy de PyTorch.")
        # Dummy model solo para que el código no se rompa si aún no subes tu clase
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(h * w, components),
            nn.ReLU(),
            nn.Linear(components, n_classes)
        )
        
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    return model, criterion, optimizer
