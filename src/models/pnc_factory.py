import torch
import torch.nn as nn
import torch.optim as optim

def build_pnc(config, device):
    """
    Fábrica para instanciar el Probabilistic Neural Circuit (PNC) de Zuidberg.
    """
    print(f"[*] Construyendo PNC real (PyTorch) para {config['dataset'].upper()}...")
    
    # Extraer dimensiones espaciales y arquitectónicas
    h = config['arquitectura']['height']
    w = config['arquitectura']['width']
    n_classes = config['arquitectura']['n_classes']
    components = config['arquitectura'].get('components', 10)
    mixing = config['arquitectura'].get('mixing', 'sum')
    
    # Hiperparámetros de optimización
    lr = config['entrenamiento'].get('lr', 0.01)
    momentum = config['entrenamiento'].get('momentum', 0.9)

    # Importamos desde la librería clonada de Zuidberg
    try:
        from circuits.pncrc import GenDisPNCRC
        model = GenDisPNCRC(
            height=h, 
            width=w, 
            components=components,
            n_classes=n_classes, 
            mixing=mixing
        ).to(device)
    except ImportError as e:
        raise ImportError(f"No se encontró la librería 'circuits'. Asegúrate de haber clonado ProbabilisticNeuralCircuits. Detalle: {e}")
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    return model, criterion, optimizer
