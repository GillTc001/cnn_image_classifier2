from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, x_test, y_test, class_names):
    # Model-Prediction
    preds = model.predict(x_test).argmax(axis=1)
    print(classification_report(y_test, preds, target_names=class_names))
    
    # Erstellung Confusion_Matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()