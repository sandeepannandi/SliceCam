import cv2
import mediapipe as mp
import numpy as np
import pygame
import random
import math
import os
from pygame import mixer
from collections import deque

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

pygame.init()
mixer.init()

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
FRUIT_TYPES = ['apple', 'mango', 'watermelon', 'grapes']
FRUIT_SIZE = 100  # Size of fruit images
SPAWN_WIDTH = 400  # Width of spawn area in the middle
SPAWN_X_MIN = (WINDOW_WIDTH - SPAWN_WIDTH) // 2
SPAWN_X_MAX = SPAWN_X_MIN + SPAWN_WIDTH

#fruit images
def load_fruit_images():
    fruit_images = {}
    # Load regular fruits
    for fruit in FRUIT_TYPES:
        image_path = os.path.join('images', f'{fruit}.png')
        if os.path.exists(image_path):
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            # Resize image while maintaining aspect ratio
            aspect_ratio = img.shape[1] / img.shape[0]
            new_height = FRUIT_SIZE
            new_width = int(new_height * aspect_ratio)
            img = cv2.resize(img, (new_width, new_height))
            fruit_images[fruit] = img
        else:
            print(f"Warning: {image_path} not found. Using colored circle instead.")
            img = np.zeros((FRUIT_SIZE, FRUIT_SIZE, 4), dtype=np.uint8)
            color = {
                'apple': (0, 0, 255, 255),      # Red
                'mango': (0, 165, 255, 255),    # Orange
                'watermelon': (0, 255, 0, 255), # Green
                'grapes': (128, 0, 128, 255)    # Purple
            }[fruit]
            cv2.circle(img, (FRUIT_SIZE//2, FRUIT_SIZE//2), FRUIT_SIZE//2, color, -1)
            fruit_images[fruit] = img
    
    bomb_path = os.path.join('images', 'bomb.png')
    if os.path.exists(bomb_path):
        img = cv2.imread(bomb_path, cv2.IMREAD_UNCHANGED)
        aspect_ratio = img.shape[1] / img.shape[0]
        new_height = FRUIT_SIZE
        new_width = int(new_height * aspect_ratio)
        img = cv2.resize(img, (new_width, new_height))
        fruit_images['bomb'] = img
    else:
        print("Warning: bomb.png not found. Using colored circle instead.")
        img = np.zeros((FRUIT_SIZE, FRUIT_SIZE, 4), dtype=np.uint8)
        cv2.circle(img, (FRUIT_SIZE//2, FRUIT_SIZE//2), FRUIT_SIZE//2, (0, 0, 0, 255), -1)
        fruit_images['bomb'] = img
    
    return fruit_images

class Fruit:
    def __init__(self, fruit_images):
        # 5% chance of spawning a bomb
        if random.random() < 0.05:
            self.type = 'bomb'
        else:
            self.type = random.choice(FRUIT_TYPES)
        
        self.image = fruit_images[self.type]
        self.radius = FRUIT_SIZE // 2
        self.x = random.randint(SPAWN_X_MIN + self.radius, SPAWN_X_MAX - self.radius)
        self.y = WINDOW_HEIGHT + self.radius
        self.speed = random.randint(8, 12)
        # Reduced angle range for more vertical movement
        self.angle = random.uniform(-math.pi/8, math.pi/8)
        self.sliced = False

    def update(self):
        if not self.sliced:
            self.x += math.sin(self.angle) * self.speed
            self.y -= self.speed

    def draw(self, frame):
        if not self.sliced:
            # Calculate the position to place the image
            x1 = int(self.x - self.image.shape[1] // 2)
            y1 = int(self.y - self.image.shape[0] // 2)
            x2 = x1 + self.image.shape[1]
            y2 = y1 + self.image.shape[0]

            # Check if the image is within frame bounds
            if (x1 < WINDOW_WIDTH and x2 > 0 and y1 < WINDOW_HEIGHT and y2 > 0):
                # Create a region of interest (ROI) for the image
                roi = frame[max(0, y1):min(WINDOW_HEIGHT, y2), max(0, x1):min(WINDOW_WIDTH, x2)]
                
                img_roi = self.image[max(0, -y1):min(self.image.shape[0], WINDOW_HEIGHT-y1),
                                   max(0, -x1):min(self.image.shape[1], WINDOW_WIDTH-x1)]
                
                if img_roi.shape[2] == 4:
                    bgr = img_roi[:, :, :3]
                    alpha = img_roi[:, :, 3] / 255.0
                    
                    for c in range(3):
                        roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * bgr[:, :, c]
                else:
                    roi[:] = img_roi

    def slice(self, slice_line):
        if not self.sliced:
            self.sliced = True
            return True
        return False

def intersects_circle(p1, p2, center, radius):
    # p1 = (x1, y1), p2 = (x2, y2) are points on the line segment
    # center = (cx, cy) is the circle center
    # radius = r

    x1, y1 = p1
    x2, y2 = p2
    cx, cy = center
    r = radius

    dx, dy = x2 - x1, y2 - y1
    # Vector from p1 to circle center
    fx, fy = cx - x1, cy - y1

    # Project vector p1-center onto vector p1-p2
    # t is the parameter along the line segment
    dot_product = fx * dx + fy * dy
    len_sq = dx * dx + dy * dy
    t = dot_product / len_sq if len_sq != 0 else 0

    # Find the point on the line closest to the circle center
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    # This uses a small tolerance for floating point comparisons
    on_segment = (min(x1, x2) - 0.1 <= closest_x <= max(x1, x2) + 0.1 and
                  min(y1, y2) - 0.1 <= closest_y <= max(y1, y2) + 0.1)

    dist_sq = (closest_x - cx)**2 + (closest_y - cy)**2

    # If the closest point is on the segment and its distance to the center is <= radius, it intersects
    if on_segment and dist_sq <= r**2:
        return True

    # If the closest point is not on the segment, check distance to endpoints
    dist_p1_sq = (x1 - cx)**2 + (y1 - cy)**2
    dist_p2_sq = (x2 - cx)**2 + (y2 - cy)**2

    return dist_p1_sq <= r**2 or dist_p2_sq <= r**2

def main():
    # Load fruit images
    fruit_images = load_fruit_images()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)

    fruits = []
    score = 0
    last_hand_pos = None
    slice_line = None
    game_over = False
    
    # Create a deque to store recent hand positions for the trail effect
    hand_positions = deque(maxlen=10)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Draw hand landmarks and track hand movement
        current_pos = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get index finger tip position
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                current_pos = (int(index_tip.x * WINDOW_WIDTH), int(index_tip.y * WINDOW_HEIGHT))
                
                # Add current position to the trail
                hand_positions.append(current_pos)
                
                if last_hand_pos:
                    slice_line = (last_hand_pos, current_pos)
                    # Draw the main slice line
                    cv2.line(frame, last_hand_pos, current_pos, (255, 255, 255), 3)
                
                last_hand_pos = current_pos
                
                # Draw the trail effect
                for i in range(len(hand_positions) - 1):
                    alpha = 1 - (i / len(hand_positions))
                    color = (int(255 * alpha), int(255 * alpha), int(255 * alpha))
                    thickness = max(1, int(3 * alpha))
                    cv2.line(frame, hand_positions[i], hand_positions[i + 1], color, thickness)
        else:
            last_hand_pos = None
            slice_line = None
            hand_positions.clear()

        # Spawn new fruits
        if random.random() < 0.01:
            if len(fruits) < 5:
                fruits.append(Fruit(fruit_images))

        # Update and draw fruits
        for fruit in fruits[:]:
            fruit.update()
            fruit.draw(frame)

            # Check for slicing
            if slice_line and not fruit.sliced:
                if intersects_circle(slice_line[0], slice_line[1], (int(fruit.x), int(fruit.y)), fruit.radius):
                    if fruit.slice(slice_line):
                        if fruit.type == 'bomb':
                            game_over = True
                        else:
                            score += 10

            # Remove fruits that are off screen or sliced
            if fruit.y < -fruit.radius or fruit.sliced:
                fruits.remove(fruit)

        # Draw score
        cv2.putText(frame, f'Score: {score}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Game over screen
        if game_over:
            cv2.putText(frame, 'GAME OVER', (WINDOW_WIDTH//2 - 150, WINDOW_HEIGHT//2 - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            cv2.putText(frame, f'Final Score: {score}', (WINDOW_WIDTH//2 - 150, WINDOW_HEIGHT//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, 'Press R to Restart', (WINDOW_WIDTH//2 - 150, WINDOW_HEIGHT//2 + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display the frame
        cv2.imshow('Fruit Ninja', frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r') and game_over:
            fruits = []
            score = 0
            game_over = False
            hand_positions.clear()
        elif key != 0xFF:
            if not game_over:
                last_hand_pos = None
                slice_line = None
                hand_positions.clear()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
