import tensorflow as tf
import numpy as np
import os
from model_builder import create_model

# CONFIG
EPOCHS = 40
BATCH_SIZE = 32
BASE_DIR = r'C:\Users\aadir\Project_Instrument'

def train():
    # 1. Load Data
    print("Loading data for STRICT training...")
    train_data = np.load(os.path.join(BASE_DIR, 'data', 'processed', 'train.npz'))
    test_data = np.load(os.path.join(BASE_DIR, 'data', 'processed', 'test.npz'))
    
    X_train, Y_train = train_data['X'], train_data['Y']
    X_test, Y_test = test_data['X'], test_data['Y']

    # --- THE FIX: STRICT LEARNING ---
    # We are using the raw labels. In OpenMIC, '0' often means "Unknown".
    # By training on '0' directly, we force the model to treat "Unknown" as "Absent".
    # This kills the "50% guess" behavior.
    print("Applying 'Weak Negative' assumption (Unknown = Absent)...")
    
    # 2. Build Model
    model = create_model()
    
    # 3. Compile
    # We use a standard learning rate.
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['AUC', 'binary_accuracy'])

    # 4. Callbacks for intelligence
    callbacks = [
        # Slow down if we get stuck
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_auc', factor=0.5, patience=3, mode='max', verbose=1
        ),
        # Stop if we stop improving
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc', patience=8, mode='max', restore_best_weights=True, verbose=1
        )
    ]

    # 5. Train
    print("\nStarting Strict Training...")
    model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # 6. Save
    save_path = os.path.join(BASE_DIR, 'models', 'instrument_classifier.h5')
    model.save(save_path)
    print(f"\nSUCCESS! Strict model saved to: {save_path}")

if __name__ == "__main__":
    train()