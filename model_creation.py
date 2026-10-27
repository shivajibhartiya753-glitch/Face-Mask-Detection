import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model

def build_mask_detector():
    # Load MobileNetV2 without the top classification layer
    base_model = MobileNetV2(weights="imagenet", include_top=False, 
                             input_tensor=Input(shape=(224, 224, 3)))
    
    # Construct the head of the model
    head_model = base_model.output
    head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
    head_model = Flatten(name="flatten")(head_model)
    head_model = Dense(128, activation="relu")(head_model)
    head_model = Dropout(0.5)(head_model)
    head_model = Dense(3, activation="softmax")(head_model)
    
    # Combine base and head
    model = Model(inputs=base_model.input, outputs=head_model)
    
    # Freeze the base layers so they aren't updated during the first training pass
    for layer in base_model.layers:
        layer.trainable = False
        
    # Compile the model
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

model = build_mask_detector()
model.summary()