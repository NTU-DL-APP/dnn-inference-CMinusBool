import tensorflow as tf
from tensorflow import keras
import numpy as np
import json


def train_and_save_model():
    """
    Trains a neural network on the Fashion-MNIST dataset and saves it.
    """
    # 1. Load and preprocess the dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    print("Data loaded and preprocessed.")
    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")

    # 2. Design the model
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28), name="flatten_layer"),
        keras.layers.Dense(128, activation="relu", name="dense_layer_1"),
        keras.layers.Dense(10, activation="softmax", name="output_layer")
    ])

    model.summary()

    # 3. Compile the model
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 4. Train the model
    print("\nStarting model training...")
    history = model.fit(
        x_train,
        y_train,
        epochs=15,
        validation_split=0.1,
        verbose=2
    )
    print("Model training complete.")

    # 5. Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"\nTest accuracy: {test_acc:.4f}")

    # 6. Save the trained model
    model_filename = "fashion_mnist_model.h5"
    model.save(model_filename)
    print(f"\nModel successfully saved to {model_filename}")


def export_for_numpy_inference(model_path="fashion_mnist_model.h5"):
    """
    Loads a saved Keras .h5 model and extracts its architecture and weights
    into the format required by the nn_predict.py script.

    Args:
        model_path (str): Path to the saved .h5 model file.

    Returns:
        tuple: A tuple containing (model_arch, weights_dict)
    """
    print(f"\nExporting model from {model_path} for NumPy inference...")

    try:
        model = keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

    model_arch = []
    weights_dict = {}

    for layer in model.layers:
        layer_details = {
            "name": layer.name,
            "type": layer.__class__.__name__,
            "config": layer.get_config(),
            "weights": []
        }

        # Get layer weights and add them to the weights dictionary
        layer_weights = layer.get_weights()
        if layer_weights:
            # For Dense layers, weights are [Kernel, Bias]
            w_name = f"{layer.name}_W"
            b_name = f"{layer.name}_b"
            weights_dict[w_name] = layer_weights[0]
            weights_dict[b_name] = layer_weights[1]
            layer_details["weights"] = [w_name, b_name]

        model_arch.append(layer_details)

    print("Model architecture and weights extracted successfully.")

    # You can now use `model_arch` and `weights_dict` with your nn_inference function.
    # For example, you could save them to disk:
    # with open('model_arch.json', 'w') as f:
    #     json.dump(model_arch, f, indent=4)
    # np.savez('model_weights.npz', **weights_dict)

    return model_arch, weights_dict


if __name__ == '__main__':
    # Step 1: Train and save the model
    train_and_save_model()

    # Step 2: Export the saved model for the NumPy inference script
    # This demonstrates that the saved model is compatible.
    model_arch, weights = export_for_numpy_inference()

    if model_arch and weights:
        print("\n--- Verification ---")
        print("Model Architecture for nn_predict.py:")
        print(json.dumps(model_arch, indent=2))

        print("\nExtracted Weight Keys for nn_predict.py:")
        print(list(weights.keys()))
        print("--------------------")