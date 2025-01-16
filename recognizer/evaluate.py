import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, testX, testY) -> None:
    predictions = model.predict(testX)
    predictions = np.argmax(predictions, axis=1)
    testY = np.argmax(testY, axis=1)

    conf_matrix = confusion_matrix(testY, predictions)

    class_report = classification_report(testY, predictions)
    print("Classification Report:")
    print(class_report)

    print("Confusion Matrix:")
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()