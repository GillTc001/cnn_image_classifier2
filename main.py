from src.dataloader import load_data
from src.train import train_model
from src.evaluate import evaluate_model

# Class_Names Definition, label von 0 bis 9
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Empfangen von Training- und Testset data
x_train, y_train, x_test, y_test = load_data()

# Model trainieren (Funktion wird aufgerufen) und Daten Übergabe

# x_train, y_train, x_val, y_val, input_shape, num_classes. y_test und x_test werden limitiert. sonst soll 10000 werte sein
model = train_model(x_train, y_train, x_test[:5000], y_test[:5000], x_train.shape[1:], 10)

# Model Evaluieren
evaluate_model(model, x_test[5000:], y_test[5000:], class_names)