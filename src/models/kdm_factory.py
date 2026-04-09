# Archivo: src/models/kdm_factory.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from kdm.models.kdm_class_model import KDMClassModel

def build_kdm_model(config, x_train_fold, y_train_fold, input_shape=(784,), num_classes=10):
    """
    Construye e inicializa un Kernel Density Matrix Classification Model.
    """
    # 1. Definición del Encoder Subyacente
    encoder = Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(256, activation='relu'),
        layers.Dense(config['encoded_size'], activation='relu')
    ], name="kdm_encoder")

    # 2. Instanciación del KDM
    modelo_kdm = KDMClassModel(
        encoded_size=config['encoded_size'],
        dim_y=num_classes,
        encoder=encoder,
        n_comp=config['n_comp'],
        sigma=0.5, # Idealmente, esto también debería ser un hiperparámetro
        sigma_trainable=True
    )

    # Nota Teórica: En clasificación, Sparse Categorical Crossentropy es equivalente
    # al Negative Log-Likelihood (NLL) del modelo condicional P(Y|X).
    modelo_kdm.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config['lr']),
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )

    # 3. Inicialización de Componentes (Requisito estricto de KDM)
    samples_x = x_train_fold[:config['n_comp']]
    samples_y_sparse = y_train_fold[:config['n_comp']]
    samples_y_onehot = keras.utils.to_categorical(samples_y_sparse, num_classes=num_classes)

    modelo_kdm.init_components(samples_x, samples_y_onehot, init_sigma=True)
    
    # 4. Cálculo de Parámetros
    total_params = np.sum([keras.backend.count_params(w) for w in modelo_kdm.trainable_weights])

    return modelo_kdm, total_params
