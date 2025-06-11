import numpy as np
from keras.preprocessing import image
from src.model import build_cnn


# Tester sur une image personnalisée (optionnel)

def predict_image(img_path, model_path, class_names):
    img = image.load_img(img_path, target_size=(32, 32))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    model = build_cnn((32, 32, 3), len(class_names))
    model.load_weights(model_path)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    print(f"Prédiction : {predicted_class}")