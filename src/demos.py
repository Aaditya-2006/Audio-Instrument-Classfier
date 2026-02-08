import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa
import os
import tkinter as tk
from tkinter import filedialog

# CONFIG
MODEL_PATH = r'C:\Users\aadir\Project_Instrument\models\instrument_classifier.h5'
INSTRUMENTS = ['accordion', 'banjo', 'bass', 'cello', 'clarinet', 'cymbal', 'drums', 'flute', 'guitar', 'mallet_percussion', 'mandolin', 'organ', 'piano', 'saxophone', 'synthesizer', 'trombone', 'trumpet', 'ukulele', 'violin', 'voice']

def predict_custom_audio(file_path):
    print(f"\nProcessing: {os.path.basename(file_path)}")
    
    try:
        vgg_model = hub.load('https://tfhub.dev/google/vggish/1')
        audio, sr = librosa.load(file_path, sr=16000, mono=True)
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        return

    # 1. Extract Features (This gives a sequence of 128-D vectors)
    # If your song is 10 seconds, embeddings shape will be roughly [10, 128]
    embeddings = vgg_model(audio).numpy()
    
    # 2. Normalization (Matches your /255 training logic)
    # We map the standard VGGish range (-1 to +1) to (0 to 1)
    embeddings = (embeddings + 1.0) / 2.0
    embeddings = np.clip(embeddings, 0.0, 1.0)

    # 3. SLIDING WINDOW PREDICTION
    # Instead of averaging first, we predict on EVERY 1-second chunk
    print("AI is scanning the timeline...")
    chunk_predictions = model.predict(embeddings) # Shape: [Num_Seconds, 20_Instruments]

    # 4. AGGREGATE RESULTS (Max Pooling)
    # We take the MAX confidence found for each instrument across the whole song.
    # If a guitar plays for even 2 seconds, this will catch it!
    final_prediction = np.max(chunk_predictions, axis=0)

    # 5. Show Results
    print("\n" + "="*30)
    print(f" RESULTS FOR: {os.path.basename(file_path)}")
    print("="*30)
    
    found = False
    # Sort results by confidence
    sorted_indices = np.argsort(final_prediction)[::-1]
    
    for i in sorted_indices:
        prob = final_prediction[i]
        # Using 0.60 to ensure we only show "confident" hits
        if prob > 0.60: 
            print(f"  [+] {INSTRUMENTS[i].upper()}: {prob*100:.1f}%")
            found = Truew
            
    if not found:
        print("  [-] No instruments detected with high confidence.")

if __name__ == "__main__":
    print("Select an audio file...")
    root = tk.Tk()
    root.withdraw() 
    file_path = filedialog.askopenfilename(title="Select Audio File")
    
    if file_path:
        predict_custom_audio(file_path)