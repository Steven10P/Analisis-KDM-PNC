# Archivo: src/models/kdm_factory_v2.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from kdm.models.kdm_class_model import KDMClassModel

def build_kdm_model_v2(config, x_train_fold, y_train_fold):
    """
    Versión Evolucionada (V2) para SVHN y Datasets Multicanal.
    Detecta dimensiones automáticamente para evitar ValueErrors.
    """
    # 1. Inferencia de dimensiones
    input_dim = x_train_fold.shape[1] 
    
    if len(y_train_fold.shape) > 1:
        num_classes = y_train_fold.shape[1]
        y_train_onehot = y_train_fold
    else:
        num_classes = len(np.unique(y_train_fold))
        y_train_onehot = keras.utils.to_categorical(y_train_fold, num_classes=num_classes)

    # 2. Arquitectura Dinámica
    encoder = Sequential([
        layers.Input(shape=(input_dim,)), 
        layers.Dense(512, activation='relu'), # Aumentamos capacidad para SVHN
        layers.Dense(config['encoded_size'], activation='relu')
    ], name="kdm_encoder_v2")

    # 3. Instanciación
    modelo_kdm = KDMClassModel(
        encoded_size=config['encoded_size'],
        dim_y=num_classes,
        encoder=encoder,
        n_comp=config['n_comp'],
        sigma=config.get('sigma', 0.5), 
        sigma_trainable=True
    )

    modelo_kdm.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config['lr']),
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )

    # 4. Inicialización con muestras reales
    modelo_kdm.init_components(x_train_fold[:config['n_comp']], 
                               y_train_onehot[:config['n_comp']], 
                               init_sigma=True)
    
    total_params = np.sum([keras.backend.count_params(w) for w in modelo_kdm.trainable_weights])
    return modelo_kdm, total_params