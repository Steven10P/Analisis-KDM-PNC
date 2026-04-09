import tensorflow as tf
from tensorflow.keras import layers

def build_kdm(config):
    """
    Fábrica para construir y compilar el Kernel Density Model (KDM) real.
    """
    print(f"[*] Construyendo KDM REAL (TensorFlow) para {config['dataset'].upper()}...")
    
    # 1. Leer hiperparámetros desde el YAML (config)
    input_dim = config['arquitectura']['input_dim']
    n_classes = config['arquitectura']['n_classes']
    sigma = config['arquitectura'].get('sigma', 0.5)
    
    n_comp = config.get('n_comp', 256)
    encoded_size = config.get('encoded_size', 64)
    lr = config['entrenamiento'].get('learning_rate', 0.001)

    # 2. Importar la clase real desde el repositorio clonado
    try:
        from kdm.models.kdm_class_model import KDMClassModel
    except ImportError as e:
        raise ImportError(f"No se encontró la librería 'kdm'. Asegúrate de haberla instalado. Detalle: {e}")

    # 3. Construir el Encoder (Típicamente necesario para el KDMClassModel)
    encoder = tf.keras.Sequential([
        layers.InputLayer(input_shape=(input_dim,)),
        layers.Dense(encoded_size, activation='relu')
    ], name="kdm_encoder")

    # 4. Instanciar el Modelo KDM Real
    model = KDMClassModel(
        n_comp=n_comp,
        encoder=encoder,
        num_classes=n_classes,
        sigma=sigma
    )
    
    # 5. Compilación
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
