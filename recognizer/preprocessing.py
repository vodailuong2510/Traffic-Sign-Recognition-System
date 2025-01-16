from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer

def preprocess_data(trainX, trainY, validX, validY, testX, testY):
    trainX, trainY = shuffle(trainX, trainY, random_state=0)
    
    trainX = trainX.astype("float")/255.0
    validX = validX.astype("float")/255.0
    testX = testX.astype("float")/255.0

    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    validY = lb.fit_transform(validY)
    testY = lb.fit_transform(testY)

    return trainX, trainY, validX, validY, testX, testY