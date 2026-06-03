# SignLang AI

A real-time sign language recognition system that watches your hand gestures through a webcam and speaks back what it sees. Show a gesture, get the word spoken aloud in under a second.

## What It Does

SignLang AI listens to your hands. Point your webcam at yourself, make a sign, and it'll tell you (or whoever's listening) what you just signed — no prior knowledge of sign language needed. It tracks 21 different points on each hand, runs them through a neural network, and converts the gesture into audio.

The whole process happens live on your machine. Latency is under a second, confidence scores are displayed in real-time, and there's a dashboard that shows exactly what the model is seeing and thinking.

## Why This Exists

Communication through sign language usually means both people need to know it. SignLang AI breaks that barrier — it's a bridge between signed and spoken words.

## The Numbers

- **99.4%** validation accuracy
- **15** gesture classes (yes, no, hello, help, emergency, and more)
- **4,500+** training samples
- **10-frame** smoothing buffer to kill false positives
- **<1 second** latency from gesture to speech

## How It Works

### 1. Capture
Your webcam feed streams in real-time. MediaPipe detects both hands and your full body pose every single frame.

### 2. Normalize
The system grabs 21 landmarks from each hand and normalizes them to your wrist position and hand size. This means the same gesture looks the same to the model whether your hand is close to the camera or far away.

### 3. Inference
A dense neural network scores those 63 normalized values against all 15 gesture classes and returns a confidence score for each one.

### 4. Smooth & Speak
A 10-frame majority vote buffer eliminates jitter. Once confidence stays above 85% across that buffer, the word gets spoken aloud through your speaker.

## Features

- 15 gesture classes trained and ready to go
- Real-time body-relative position tracking (shoulder and nose landmarks give spatial context)
- 10-frame prediction smoothing — no more flickering between similar gestures
- Text-to-speech output with pyttsx3
- Live dashboard showing hand skeleton, normalized landmarks, class probabilities, and the entire pipeline
- Handles one or two hands at the same time
- Under-a-second response time

## The Dashboard

Three panels show exactly what's happening inside the model:

**Left Panel** — Hand skeleton and normalized landmarks. Shows the isolated hand from the first detected hand's 21 points, redrawn cleanly. On the right, a bar graph displays the first 14 normalized hand values plus 6 body position values calculated relative to your nose and shoulders.

**Middle Panel** — Pipeline view on a black background showing exactly what MediaPipe is tracking. Body pose in cyan, hand connections in orange. Fingertips (landmarks 4, 8, 12, 16, 20) get emphasized with layered circles. A dotted line connects your wrist to your nose for spatial reference. Small stats in the corner confirm hand count and tracking status.

**Right Panel** — Class probabilities as a bar chart updated every frame. Your current prediction with confidence bar. Below that, a voting history from the last 10 frames — you can literally watch the decision form in real-time. Only speaks once confidence holds above 85%.

## What I Learned Building This

**Landmark normalization was everything.** Hand size and distance kept throwing off predictions. Once I normalized every landmark relative to the wrist and overall hand size, the same gesture stayed consistent whether the hand was 6 inches or 2 feet from the camera.

**Prediction flicker was brutal without smoothing.** Frame-to-frame predictions jumped between similar classes. A 10-frame majority vote fixed it without adding noticeable lag.

**Hand slot assignment is trickier than it sounds.** MediaPipe doesn't guarantee which hand is hand 1 vs hand 2 between frames. I built consistent slot assignment so each hand keeps its identity across the whole sequence.

**Body position matters.** The same handshape means different things at your chest vs your head. Pose landmarks feed the model that spatial context so it can tell signs apart.

## Tech Stack

- **Python** — core language
- **TensorFlow** — the neural network
- **MediaPipe** — hand and body landmark detection
- **OpenCV** — video capture and dashboard rendering
- **pyttsx3** — text-to-speech

Model trained locally on self-collected gesture data. Everything runs in a single real-time loop at your webcam's framerate.

## Getting Started

```bash
# Install dependencies
pip install tensorflow mediapipe opencv-python pyttsx3

# Run the system
python main.py
```

Point your webcam at yourself and start signing. The dashboard will pop up and start tracking. Once you hit 85% confidence on a gesture, you'll hear it spoken back.

## Try It Out

There's a live demo you can test. Check the gesture classes, watch the confidence scores, see the dashboard in action.

---

That's it. Sign a gesture, get a word back. No setup hassle, no delays.
