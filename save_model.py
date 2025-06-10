# export_model.py
import os
import json
import numpy as np
import tensorflow as tf

YOUR_MODEL_NAME = 'fashion_mnist'
MODEL_DIR = 'model'
os.makedirs(MODEL_DIR, exist_ok=True)

# final paths
MODEL_WEIGHTS_PATH = f'{MODEL_DIR}/{YOUR_MODEL_NAME}.npz'
MODEL_ARCH_PATH    = f'{MODEL_DIR}/{YOUR_MODEL_NAME}.json'

# load Keras model
model = tf.keras.models.load_model(f'{MODEL_DIR}/{YOUR_MODEL_NAME}.h5')

# extract & save weights
params = {}
for layer in model.layers:
    for i, w in enumerate(layer.get_weights()):
        params[f'{layer.name}_{i}'] = w
np.savez(MODEL_WEIGHTS_PATH, **params)

# extract & save architecture
arch = []
for layer in model.layers:
    arch.append({
        "name": layer.name,
        "type": layer.__class__.__name__,
        "config": layer.get_config(),
        "weights": [f"{layer.name}_{i}" for i in range(len(layer.get_weights()))]
    })
with open(MODEL_ARCH_PATH, 'w') as f:
    json.dump(arch, f, indent=2)