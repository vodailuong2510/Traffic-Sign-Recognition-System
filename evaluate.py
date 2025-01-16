from tensorflow.models import load_model
from recognizer.preprocessing import preprocess_data
from recognizer.utils import load_data
from recognizer.evaluate import evaluate_model

train_path = "./data/train.p"
valid_path = "./data/valid.p"
test_path = "./data/test.p"
_, _, _, _, testX, testY = preprocess_data(load_data("./data/train.p", "./data/valid.p", "./data/test.p"))

model_path= "TrafficSign_Recognizer.h5"
model = load_model(model_path)

evaluate_model(model, testX, testY)
