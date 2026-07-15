import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import urllib.request
import time

# Hand skeleton connections for drawing manually
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky and palm
]

class HandTracker:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7, debug=False):
        self.debug = debug
        
        # The new MediaPipe Tasks API requires a model asset file.
        model_path = 'hand_landmarker.task'
        if not os.path.exists(model_path):
            print(f"Downloading MediaPipe Hand Landmarker model to {model_path}...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            print("Download complete.")

        # Initialize the new HandLandmarker from tasks API
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=vision.RunningMode.IMAGE
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # Drawing state
        self.canvas = None
        self.xp, self.yp = 0, 0
        self.draw_color = (0, 255, 0)  # Green color in BGR
        self.draw_thickness = 5
        
        self.writing_start_time = None

    def process_frame(self, img):
        # Flip the image horizontally for a selfie-view display
        img = cv2.flip(img, 1)
        
        # Initialize canvas if it hasn't been created yet
        if self.canvas is None or self.canvas.shape != img.shape:
            self.canvas = np.zeros_like(img)

        # Convert the BGR image to RGB for MediaPipe processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to MediaPipe Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        # Detect hands
        detection_result = self.detector.detect(mp_image)
        
        current_time = time.time()
        eraser_rects = []
        
        if detection_result.hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
                # Determine handedness
                handedness = detection_result.handedness[hand_idx][0].category_name
                
                # Extract landmark coordinates
                h, w, c = img.shape
                lm_list = []
                for id, lm in enumerate(hand_landmarks):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append([id, cx, cy])
                
                if self.debug and len(lm_list) > 0:
                    # Draw hand skeleton manually
                    for connection in HAND_CONNECTIONS:
                        idx1, idx2 = connection
                        pt1 = (lm_list[idx1][1], lm_list[idx1][2])
                        pt2 = (lm_list[idx2][1], lm_list[idx2][2])
                        cv2.line(img, pt1, pt2, (255, 0, 0), 2)  # Blue lines
                    for lm in lm_list:
                        cv2.circle(img, (lm[1], lm[2]), 4, (0, 0, 255), cv2.FILLED) # Red joints
                
                if len(lm_list) != 0:
                    if handedness == "Left":  # This means physical Right hand (Drawing)
                        # Index finger tip is landmark 8
                        x1, y1 = lm_list[8][1], lm_list[8][2]
                        # Middle finger tip is landmark 12
                        x2, y2 = lm_list[12][1], lm_list[12][2]
                        
                        # Check which fingers are up
                        fingers = []
                        
                        # Thumb (rough check comparing x-coordinates)
                        if lm_list[4][1] < lm_list[3][1]: fingers.append(1)
                        else: fingers.append(0)
                            
                        # 4 Fingers (checking y-coordinates of tips vs lower joints)
                        tip_ids = [8, 12, 16, 20]
                        for id in tip_ids:
                            if lm_list[id][2] < lm_list[id - 2][2]: fingers.append(1)
                            else: fingers.append(0)
                                
                        # Drawing mode: Index finger is up (fingers[1] == 1), Middle finger is down (fingers[2] == 0)
                        if fingers[1] == 1 and fingers[2] == 0:
                            if self.writing_start_time is None:
                                self.writing_start_time = current_time
                                self.xp, self.yp = 0, 0
                                
                            # Check if 0.5 seconds have passed
                            if current_time - self.writing_start_time > 0.5:
                                cv2.circle(img, (x1, y1), 15, self.draw_color, cv2.FILLED)
                                
                                # If starting to draw, set xp, yp to current position
                                if self.xp == 0 and self.yp == 0:
                                    self.xp, self.yp = x1, y1
                                    
                                # Draw a line from previous point to current point on the canvas
                                cv2.line(self.canvas, (self.xp, self.yp), (x1, y1), self.draw_color, self.draw_thickness)
                                
                                # Update previous point
                                self.xp, self.yp = x1, y1
                            else:
                                # Show a yellow circle indicating it's waiting
                                cv2.circle(img, (x1, y1), 15, (0, 255, 255), 2)
                        else:
                            # Not in drawing mode, reset timer and points
                            self.writing_start_time = None
                            self.xp, self.yp = 0, 0
                            
                    elif handedness == "Right":  # This means physical Left hand (Eraser)
                        # Check if it is a closed fist
                        fingers_up = 0
                        tip_ids = [8, 12, 16, 20]
                        for id in tip_ids:
                            if lm_list[id][2] < lm_list[id - 2][2]: fingers_up += 1
                                
                        # If all 4 fingers are folded into the palm
                        if fingers_up == 0:
                            x_coords = [lm[1] for lm in lm_list]
                            y_coords = [lm[2] for lm in lm_list]
                            # Define bounding box for the fist (with 20px padding)
                            ex1, ey1 = max(0, min(x_coords) - 20), max(0, min(y_coords) - 20)
                            ex2, ey2 = min(w, max(x_coords) + 20), min(h, max(y_coords) + 20)
                            eraser_rects.append((ex1, ey1, ex2, ey2))
                            
                            # Draw a red rectangle showing the eraser zone
                            cv2.rectangle(img, (ex1, ey1), (ex2, ey2), (0, 0, 255), 2)

        # -------------------------------------------------------------------
        # Step 2: Stroke Grouping (Connected Components) and Selective Eraser
        # -------------------------------------------------------------------
        img_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY)
        
        # Dilate the strokes to group close components together
        # A 40x40 kernel means strokes within ~40 pixels of each other form a single group
        kernel = np.ones((40, 40), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        
        # Find contours of the grouped strokes
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            # Filter out tiny noise
            if bw > 10 and bh > 10:
                bx1, by1, bx2, by2 = x, y, x + bw, y + bh
                
                # Check if this group collides with any eraser fist
                erased = False
                for (ex1, ey1, ex2, ey2) in eraser_rects:
                    if not (ex2 < bx1 or ex1 > bx2 or ey2 < by1 or ey1 > by2):
                        erased = True
                        break
                        
                if erased:
                    # Erase this specific group from the canvas using its bloated contour
                    cv2.drawContours(self.canvas, [cnt], -1, (0, 0, 0), cv2.FILLED)
                else:
                    # Draw a bounding box around the group on the output image
                    cv2.rectangle(img, (x, y), (x+bw, y+bh), (0, 255, 0), 2)

        # -------------------------------------------------------------------
        # Merge the drawing canvas with the webcam feed
        # -------------------------------------------------------------------
        img_gray_updated = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray_updated, 50, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        
        img = cv2.bitwise_and(img, img_inv)
        img = cv2.bitwise_or(img, self.canvas)

        return img
