# SliceCam

A hand-tracking version of the classic Fruit Ninja game using OpenCV and MediaPipe.

## Requirements

- Python 3.7+
- Webcam
- Required Python packages (install using `pip install -r requirements.txt`):
  - opencv-python
  - mediapipe
  - numpy
  - pygame

## How to Run

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Run the game:
```bash
python fruit_ninja.py
```

## How to Play

1. Make sure you have good lighting for hand tracking to work properly
2. Stand in front of your webcam
3. Use your index finger to slice the fruits
4. Try to slice as many fruits as possible
5. If you miss a fruit, the game is over
6. Press 'R' to restart after game over
7. Press 'Q' to quit the game

## Features

- Real-time hand tracking using MediaPipe
- Multiple fruit types with different colors
- Slicing animation when fruits are cut
- Score tracking
- Game over screen with restart option
- Smooth fruit physics and movement 
