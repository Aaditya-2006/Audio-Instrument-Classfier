# Audio Instrument Classifier using VGGish Embeddings

## Overview
A deep learning project exploring musical instrument recognition in polyphonic audio. This project utilizes **Transfer Learning** from Google's VGGish model and classify them into 20 distinct instrument categories.

## Technical Stack
* **Language:** Python 3
* **Framework:** TensorFlow / Keras
* **Feature Extraction:** Google VGGish (VGG-like Convolutional Neural Network)
* **Signal Processing:** Scipy & Numpy (bypassing external codec dependencies like FFmpeg for robust WAV handling)

## Dataset: OpenMIC-2018
The model is trained on the **OpenMIC-2018** dataset, a large-scale benchmark for instrument recognition.
* **Content:** 20,000 audio clips, each 10 seconds long.
* **Classes:** 20 instruments including Guitar, Piano, Trumpet, Voice, Drums, and Saxophone.
* **Note:** The `data/` folder is excluded via `.gitignore` to comply with storage best practices and dataset licensing.



## Key Implementations

### 1. Robust Audio Pipeline
To ensure the model works across different Windows environments without complex codec installations, I implemented an audio loader using **Scipy**. This handles 16-bit PCM WAV data directly, performing mono-conversion and resampling to **16kHz** as required by the VGGish architecture.

### 2. Sliding Window Inference
Real-world audio is polyphonic and continuous. My `demos.py` script utilizes a **temporal sliding window** to scan the audio timeline in 1-second increments. It then applies **Max Pooling** across these chunks to detect the peak presence of instruments, which is more effective for transient sounds than global averaging.

### 3. Min-Max Normalization
To bridge the gap between training and real-world testing, I implemented a static scaling block that maps VGGish embeddings from their raw range to a normalized **[0.0, 1.0]** distribution, aligning with the training data distribution.

## Observations & Challenges
* **Timbre Confusion:** A significant finding was the spectral overlap between **high-gain electric guitars** and **brass instruments**. The model frequently misidentifies distorted guitar harmonics as Trumpet or Trombone due to similar harmonic profiles in the frequency domain.
* **Domain Shift:** While the model achieves high AUC on the OpenMIC test set, generalization to consumer-grade recordings (e.g., Radiohead's *High and Dry*) is affected by background noise, reverb, and varied recording acoustics.

## How to Run
1. **Clone the Repo:**
   ```bash
   git clone [https://github.com/Aaditya-2006/Audio-Instrument-Classfier.git](https://github.com/Aaditya-2006/Audio-Instrument-Classfier.git)
   cd Audio-Instrument-Classfier

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
3. **Run the Demo**
   ```bash
   python src/demos.py
   ```

4. ***OR train the models from scratch by running prepare_data.py and then training.py***
