
#  Emotion Detection from Voice using ML/DL

A machine learning and deep learning-based project that classifies emotions from human speech audio using extracted acoustic features. This project supports real-time predictions via a **Streamlit web app** and is trained on the **RAVDESS dataset**.

---

## Overview

This project performs emotion recognition from `.wav` audio clips using feature extraction techniques (MFCC, Mel-spectrogram, Chroma, ZCR, Bandwidth) and trains ML/DL models. The best-performing model is deployed in a user-friendly web interface.

---

## Emotions Covered

- Neutral  
- Calm  
- Happy  
- Sad  
- Angry  
- Fearful  
- Surprised  

> Note: The *Disgust* class was intentionally removed due to poor model generalization.

---

##  Dataset

- **Source**: [RAVDESS](https://zenodo.org/record/1188976)
- Contains speech and song audio samples recorded by 24 professional actors.
- Files structured as: `03-01-02-01-01-01-01.wav` (emotion, intensity, statement, repetition, actor)
- Only **preprocessed `.npy` files** are included. Raw audio should be downloaded manually and added to data folder.

---

##  Pipeline

1. **Data Loading** from pre-extracted `.npy` files
2. **Preprocessing**: Extracted features stored as NumPy arrays
3. **Feature Extraction** using:
   - 40 MFCCs
   - 125 Mel-Spectrogram values
   - Zero Crossing Rate
   - Spectral Bandwidth
   - 10 Chroma features
4. **Modeling**:
   - Final model is a Deep Neural Network trained on 179 features
5. **Deployment**:
   - Streamlit web app for real-time emotion prediction

---

##  Model Performance

| Model                             | Accuracy |
|-----------------------------------|----------|
| ANN                               | ~75%     |
| ANN(after dropping diguist class) | ~79%     |
| ANN (final)                       | **81%**  |

---

##  How to Run

### 1. Clone this repository
```bash
git clone https://github.com/yourusername/emotion-audio-app.git
cd emotion-audio-app
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App
```bash
streamlit run app.py
```

---

##  File Structure

```
emotion-audio-app/
├── app.py
├── emotion_model.pkl
├── label_encoder.pkl
├── data/
│   ├── X_ann_features.npy
│   └── y_ann_labels.npy
├── README.md
├── requirements.txt
```

---

##  Streamlit App

- Upload `.wav` file
- Audio gets processed
- Predicted emotion is displayed with probability

---

## Tech Stack

- Python 3.11
- NumPy, Pandas
- Librosa
- Scikit-learn
- Tensorflow (Keras)
- Streamlit

---

## ⚠️ Notes

- Raw audio data not uploaded due to size & licensing
- Users must manually download RAVDESS dataset if training is needed
- Gender and intensity were fixed during prediction for consistency

---

## 🙌 Acknowledgements

- [RAVDESS Dataset - Zenodo](https://zenodo.org/record/1188976)
- [Librosa](https://librosa.org/)
- [Streamlit](https://streamlit.io)

---


