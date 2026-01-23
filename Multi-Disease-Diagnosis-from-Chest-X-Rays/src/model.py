from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam

def create_model(input_shape=(224, 224, 1), num_classes=14):
    base = DenseNet121(include_top=False, weights=None, input_shape=input_shape)
    x = GlobalAveragePooling2D()(base.output)
    output = Dense(num_classes, activation='sigmoid')(x)
    model = Model(inputs=base.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model