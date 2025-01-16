import wget
import zipfile
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random

def download(link:str) -> None:
    print("Start downloading")
    wget.download(link)
    print("Download complete")

def unzip(zip_path:str, extract_path:str) -> None:
    print("Start unzipping")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Unzip complete")

def load_data(train_link:str, valid_link:str, test_link:str) -> tuple:
    with open(train_link, mode="rb") as f:
        train=pickle.load(f)
    with open(valid_link, mode="rb") as f:
        valid=pickle.load(f)
    with open(test_link, mode="rb") as f:
        test=pickle.load(f)
    
    trainX, trainY = train["features"], train["labels"]
    validX, validY = valid["features"], valid["labels"]
    testX, testY = test["features"], test["labels"]
    
    return trainX, trainY, validX, validY, testX, testY

def plot_random_images(trainX, trainY, validX, validY, testX, testY, classNames):
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    datasets = [(trainX, trainY, "Train Set"), (validX, validY, "Validation Set"), (testX, testY, "Test Set")]

    for row, (X, Y, title) in enumerate(datasets):
        indices = random.sample(range(len(X)), 5) 
        for col, idx in enumerate(indices):
            ax = axes[row, col]
            ax.imshow(X[idx]) 
            ax.set_title(classNames[Y[idx]]) 
            ax.axis("off")  

        axes[row, 0].set_ylabel(title, fontsize=14)

    plt.tight_layout()
    plt.show()