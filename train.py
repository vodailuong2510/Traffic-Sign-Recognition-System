from recognizer.utils import load_data
from recognizer.preprocessing import preprocess_data
from recognizer.model import build_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.callbacks import ModelCheckpoint

train_path = "./data/train.p"
valid_path = "./data/valid.p"
test_path = "./data/test.p"

trainX, trainY, validX, validY, _, _= preprocess_data(load_data(train_path, valid_path, test_path))
aug = ImageDataGenerator(
    rotation_range=0.18, 
    zoom_range=0.15, 
    width_shift_range=0.2, 
    height_shift_range=0.2, 
    horizontal_flip=True
) 

epochs=10
batch_size=64

model = build_model(width=32, height=32, depth=3, classes=43)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

callbacks = [
    ModelCheckpoint(
    filepath="TrafficSign_Recognizer.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
    ),
]

print("Start training")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=batch_size), validation_data=(validX, validY), 
    steps_per_epoch=trainX.shape[0]//batch_size, epochs=epochs, callbacks = callbacks, verbose=1
)
print("Training complete")

model.save("TrafficSign_Recognizer.h5")
print("Model saved")