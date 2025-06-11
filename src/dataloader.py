

from tensorflow.keras.datasets import cifar10


def load_data():
    # This is a dataset of 50,000 32x32 color training images and 10,000 test images, labeled over 10 categories.
    # Erstellung von Training- und Testset data Mithilfe von "cifar10" aus "keras.datasets"
    # ReturnValue der Funktion load_data() ist ein Tuple of NumPy arrays
    # The Pixel values range from 0 to 255.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Data Aufbereitung.
    # dataset of 50,000 32x32 color training images and 10,000 test images
    x_train, x_test = x_train / 255.0, x_test / 255.0
    # Renvoie une copie du tableau réduit en une seule dimension.
    y_train, y_test = y_train.flatten(), y_test.flatten()

    print(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")

    # Data returning
    return x_train, y_train, x_test, y_test