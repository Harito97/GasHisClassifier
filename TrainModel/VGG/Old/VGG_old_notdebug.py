import time
import math
import json
import os
import sys
import matplotlib.pyplot as plt
import keras
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D, Input, Flatten, Dropout
from keras.models import Model
from keras.optimizers import Adam, SGD
import pandas as pd
import tensorflow as tf

DATA_DIR = 'D:/Projects/GasHisClassifier/Data/data'
TARGET_SIZE = (64, 64)
BATCH_SIZE = 256

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data generators
train_datagen = ImageDataGenerator(rescale=1/255.0)
valid_datagen = ImageDataGenerator(rescale=1/255.0)
test_datagen = ImageDataGenerator(rescale=1/255.0)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'),
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

valid_generator = valid_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'valid'),
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'test'),
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Display information
print("Class Indices: ", train_generator.class_indices)
print(f"Number of training samples: {train_generator.samples}")
print(f"Number of validation samples: {valid_generator.samples}")
print(f"Number of test samples: {test_generator.samples}")

print("Training set:")
for class_name, idx in train_generator.class_indices.items():
    num_files = len(os.listdir(os.path.join(DATA_DIR, 'train', class_name)))
    print(f"{class_name} ({idx}): {num_files} files")

print("Validation set:")
for class_name, idx in valid_generator.class_indices.items():
    num_files = len(os.listdir(os.path.join(DATA_DIR, 'valid', class_name)))
    print(f"{class_name} ({idx}): {num_files} files")

print("Test set:")
for class_name, idx in test_generator.class_indices.items():
    num_files = len(os.listdir(os.path.join(DATA_DIR, 'test', class_name)))
    print(f"{class_name} ({idx}): {num_files} files")

def save_history(history):
    acc = pd.Series(history.history["accuracy"], name="accuracy")
    loss = pd.Series(history.history["loss"], name="loss")
    val_acc = pd.Series(history.history["val_accuracy"], name="val_accuracy")
    val_loss = pd.Series(history.history["val_loss"], name="val_loss")
    com = pd.concat([acc, loss, val_acc, val_loss], axis=1)
    com.to_csv("history.csv", index=False)

def plot_history(history):
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model Accuracy and Loss")
    plt.ylabel("Accuracy/Loss")
    plt.xlabel("Epoch")
    plt.legend(["accuracy", "val_accuracy", "loss", "val_loss"], loc="upper right")
    plt.show()
    plt.savefig("model_accuracy_loss.png")

start = time.process_time()
# Tính số bước cho mỗi epoch
num_train_steps = math.ceil(train_generator.samples / BATCH_SIZE)
num_valid_steps = math.ceil(valid_generator.samples / BATCH_SIZE)

classes = list(iter(train_generator.class_indices))
Inp = Input((64, 64, 3))
base_model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
x = base_model(Inp)
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(len(classes), activation="softmax")(x)
finetuned_model = Model(inputs=Inp, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = False
finetuned_model.compile(
    optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"]
)
for c in train_generator.class_indices:
    classes[train_generator.class_indices[c]] = c
finetuned_model.classes = classes
early_stopping = EarlyStopping(patience=15)
checkpointer = ModelCheckpoint(
    "vgg16_best.keras",
    verbose=1,
    save_best_only=True,
)
History = finetuned_model.fit(
    train_generator,
    steps_per_epoch=num_train_steps - 1,
    epochs=100,
    callbacks=[early_stopping, checkpointer],
    validation_data=valid_generator,
    validation_steps=num_valid_steps - 1,
)
save_history(History)
plot_history(History)
end2 = time.process_time()
print("final is in ", end2 - start)
