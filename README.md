Sign Language Translator

A real-time sign language recognition system built with Python. It uses a webcam to detect hand gestures and converts them into spoken words using text-to-speech.

MediaPipe tracks 21 landmarks on the hand every frame, normalizes them relative to wrist position and hand size, and feeds them into a trained TensorFlow neural network. A 10-frame smoothing buffer stabilizes predictions before the word is spoken aloud. Confidence scores and a live visualization dashboard show exactly what the model is seeing and how certain it is in real time.

Trained on self-collected gesture data across 15 classes including yes, no, hello, help, i love you, and more. Achieves 99%+ validation accuracy.

## Tech Stack
Python · TensorFlow · MediaPipe · OpenCV · pyttsx3
