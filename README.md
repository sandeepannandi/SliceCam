<div align="center">
  <h3 align="center">OpenCV Fruit Ninja</h3>
  <p align="center">
    A hand-tracking version of the classic Fruit Ninja game using OpenCV and MediaPipe.
    <br />
    <a href="#features"><strong>Explore the features »</strong></a>
    <br />
    <br />
    <a href="#how-to-play">View Demo</a>
    ·
    <a href="#how-to-run">Report Bug</a>
    ·
    <a href="#how-to-run">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#how-to-play">How to Play</a></li>
    <li><a href="#features">Features</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

OpenCV Fruit Ninja is an interactive, computer-vision based arcade game inspired by the classic Fruit Ninja. Instead of swiping on a touch screen, you use your actual hand movements captured by your webcam to slice fruits in real-time. The game leverages MediaPipe for accurate hand-tracking and OpenCV for rendering and computer vision operations.

### Built With

Major frameworks and libraries used to bootstrap this project:

- [![Python][Python-shield]][Python-url]
- [![OpenCV][OpenCV-shield]][OpenCV-url]
- [![MediaPipe][MediaPipe-shield]][MediaPipe-url]
- [![Pygame][Pygame-shield]][Pygame-url]

<!-- GETTING STARTED -->

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

Ensure you have Python 3.7 or higher installed on your system. You will also need a functional webcam.

### Installation

1. Clone the repository or download the source code files.
2. Install the required Python packages:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the game:
   ```sh
   python fruit_ninja.py
   ```

<!-- HOW TO PLAY -->

## How to Play

1. Make sure you are in a well-lit environment for optimal hand-tracking performance.
2. Stand in front of your webcam ensuring your hands are clearly visible.
3. Use your **index finger** to slice the fruits that appear on the screen.
4. Try to slice as many fruits as possible to achieve a high score!
5. **Watch out!** If you miss a fruit and it falls off the screen, the game is over.
6. Press `R` to restart the game after a game over.
7. Press `Q` to quit the game at any time.

<!-- FEATURES -->

## Features

- **Real-Time Hand Tracking:** Utilizes Google's MediaPipe for robust and low-latency hand tracking.
- **Dynamic Gameplay:** Multiple fruit types rendered with distinct colors and smooth physics.
- **Visual Feedback:** Satisfying slicing animations when a fruit is successfully cut.
- **Score System:** Tracks your current score to challenge yourself.
- **Intuitive Controls:** Replay or quit seamlessly directly from the game over screen.

<!-- LICENSE -->

## License

Distributed under the Apache License 2.0. See `LICENSE` for more information.

[Python-shield]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/
[OpenCV-shield]: https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white
[OpenCV-url]: https://opencv.org/
[MediaPipe-shield]: https://img.shields.io/badge/MediaPipe-00BFFF?style=for-the-badge&logo=google&logoColor=white
[MediaPipe-url]: https://mediapipe.dev/
[Pygame-shield]: https://img.shields.io/badge/Pygame-1E1E24?style=for-the-badge&logo=pygame&logoColor=white
[Pygame-url]: https://www.pygame.org/
