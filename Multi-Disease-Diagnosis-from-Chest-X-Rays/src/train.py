from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os

def train_model(model, train_gen, val_gen, epochs=5):
    #Define path to save model
    checkpoint_path = '/content/drive/MyDrive/multidisease detection/checkpoints'
    os.makedirs(checkpoint_path, exist_ok=True)

    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.2, patience=2),
        ModelCheckpoint(
            filepath=os.path.join(checkpoint_path, 'best_model.h5'),
            save_best_only=True
        )
    ]

    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks)

    #Save model after training
    final_model_path = os.path.join(checkpoint_path, 'final_model.h5')
    model.save(final_model_path)
    print(f"Final model saved at: {final_model_path}")

    return history

