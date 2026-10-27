import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model_creation import build_mask_detector

# Define paths and hyperparameters
INIT_LR = 1e-4
EPOCHS = 20
BS = 32
DATASET_PATH = "dataset" # Ensure your images are in this folder

# Initialize the data generator for data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2 # Use 20% of data for validation
)

# Load training data
train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=BS,
    class_mode="categorical",
    subset="training"
)

# Load validation data
val_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=BS,
    class_mode="categorical",
    subset="validation"
)

# Load the model architecture
print("[INFO] Compiling model...")
model = build_mask_detector()

# Train the model
print("[INFO] Training head...")
H = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BS,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BS,
    epochs=EPOCHS
)

# Save the trained model
print("[INFO] Saving mask detector model...")
model.save("mask_model.h5")