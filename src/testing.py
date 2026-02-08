import tensorflow as tf
import numpy as np
import os
import random

instruments = ['accordion', 'banjo', 'bass', 'cello', 'clarinet', 
               'cymbal', 'drums', 'flute', 'guitar', 'mallet_percussion', 
               'mandolin', 'organ', 'piano', 'saxophone', 'synthesizer', 
               'trombone', 'trumpet', 'ukulele', 'violin', 'voice']

model_path = r'C:\Users\aadir\Project_Instrument\models\instrument_classifier.h5'
test_data_path = r'C:\Users\aadir\Project_Instrument\data\processed\test.npz'

def verify():
    print("loading model")
    model = tf.keras.models.load_model(model_path)

    print("Loading test samples")
    test_data = np.load(test_data_path)
    X_test = test_data['X']
    Y_test = test_data['Y']

    idx = random.randint(0, len(X_test) - 1)
    sample = X_test[idx].reshape(1, 128)

    print("\nTesting {idx}")
    prediction = model.predict(sample)[0]

    print("\nPREDICTING")
    found_any = False
    for i, prob in enumerate(prediction):
        if prob > 0.60:
            print(f"{instruments[i].upper()}: {prob*100:.1f}%")
            found_any = True
    if not found_any:
        print("No instruments found")

    print("\nREALITY")
    for i, val in enumerate(Y_test[idx]):
        if val > 0.5:
            print(f"{instruments[i].upper()}")

if __name__ == "__main__":
    verify()