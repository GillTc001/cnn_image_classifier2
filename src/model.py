#from tensorflow.python.keras import Sequential
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout



def build_cnn(input_shape=(32, 32, 3), num_classes=10):
    # input_shape=(32, 32, 3) for 32x32 RGB pictures in data_format="channels_last"
    # Kapazität: ............, Aktivierungsfunktion für jeden Satz, 11 Eingabedaten.

    model = Sequential([
        # 4+D tensor with shape: batch_shape + (rows, cols, channels) if data_format='channels_last'.
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
        # Um die von dense erzeugten realwertigen Ausgaben in Wahrscheinlichkeiten umzuwandeln,wird die Aktivierungsfunktion 'softmax' hinzugeführt.
    ])

    return model