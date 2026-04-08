import tensorflow as tf
from tensorflow.keras import layers, models

def build_kdm(config):
    """
    Fábrica para construir y compilar el modelo Kernel Density Model (KDM).
    """
    print(f"[*] Construyendo KDM (TensorFlow) para {config['dataset'].upper()}...")
    
    # Leer hiperparámetros desde el YAML
    input_dim = config['arquitectura']['input_dim']
    n_classes = config['arquitectura']['n_classes']
    units = config['arquitectura'].get('units_dense', 128)
    lr = config['entrenamiento'].get('learning_rate', 0.001)

    # Definición de la Arquitectura
    # (Aquí puedes reemplazar esta base por tu clase custom KDM)
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(units, activation='relu')(inputs)
    outputs = layers.Dense(n_classes, activation='softmax', name='kdm_density_output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="KDM_Model")
    
    # Compilación
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
