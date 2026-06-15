# SignLang AI

Real-time sign language recognition system. Show a gesture to your webcam, get the word spoken aloud in under a second.

## What It Does

Tracks your hand gestures and converts them to speech. Uses MediaPipe to detect hand landmarks, runs them through a neural network, and speaks the result. No setup hassle.

## The Numbers

- **99.4%** validation accuracy
- **15** gesture classes 
- **4,500+** training samples
- **<1 second** latency

## How It Works

1. **Capture** — Webcam detects hands and body pose in real-time
2. **Normalize** — Hand landmarks scaled to wrist position and size
3. **Inference** — Neural network predicts the gesture
4. **Smooth & Speak** — 10-frame buffer filters noise, speaks when confident (85%+)

## Features

- 15 trained gesture classes (yes, no, hello, help, emergency, etc.)
- Real-time dashboard showing hand skeleton, landmarks, and predictions
- Works with one or two hands
- Body-relative positioning for accuracy
- Text-to-speech output via pyttsx3

## Tech Stack

Python · TensorFlow · MediaPipe · OpenCV · pyttsx3

## Quick Start

```bash
pip install tensorflow mediapipe opencv-python pyttsx3
python main.py
```

Point your webcam and start signing.

---

Sign a gesture, get a word back.
