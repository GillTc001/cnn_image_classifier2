import pandas as pd
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from src.model import build_cnn


def train_model(x_train, y_train, x_val, y_val, input_shape, num_classes):
    # Model von Type build_cnn erstellen. (aktion in src.model)
    model = build_cnn(input_shape, num_classes)

    # Nachdem wir das Modell definiert haben, kompilieren wir den Optimierer und die Verlustfunktion.
    model.compile(
        optimizer=Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    model.summary()

    # --- Early Stopping --------
    es = EarlyStopping(
        patience=5,  # how many epochs to wait before stopping
        min_delta=0.001,
        restore_best_weights=True)

    # Erstellte build_cnn Model mit fonction fit trainieren
    model.fit(x_train, y_train,
              validation_data=(x_val, y_val),
              epochs=50,
              batch_size=64,
              callbacks=[es])  # Early Stopping-callbacks in a list

    # 💾 Sauvegarder les poids
    model.save_weights("model_weights.weights.h5")

    # Model Rückgabe
    return model


