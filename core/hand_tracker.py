import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import urllib.request
import time
import math
import collections

def chaikin_smooth(points, iterations=3):
    if len(points) < 3:
        return points
    for _ in range(iterations):
        new_points = [points[0]]
        for i in range(len(points) - 1):
            p0, p1 = points[i], points[i+1]
            q = (0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1])
            r = (0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1])
            new_points.extend((q, r))
        new_points.append(points[-1])
        points = new_points
    return [(int(p[0]), int(p[1])) for p in points]

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
        
        model_path = 'hand_landmarker.task'
        if not os.path.exists(model_path):
            print(f"Downloading MediaPipe Hand Landmarker model to {model_path}...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            print("Download complete.")

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
        self.current_stroke = []
        self.exp, self.eyp = 0, 0 # Eraser points
        self.draw_color = (0, 255, 0)
        self.draw_thickness = 5
        self.eraser_thickness = 30
        
        self.writing_start_time = None
        
        # Swipe state
        self.left_x_history = collections.deque(maxlen=10)
        
        # Dragging state
        self.last_bboxes = []
        self.dragged_box = None
        self.drag_px, self.drag_py = 0, 0
        self.drag_offset_x, self.drag_offset_y = 0, 0
        
        self.thumbs_up_start_time = None
        self.thumbs_up_last_seen_time = 0
        self.last_thumb_pos = None
        self.last_saved_time = 0
        
        # Performance Cache
        self.prediction_cache = {}

        # Recognizer
        model_file = os.path.join(os.path.dirname(__file__), '..', 'training', 'models', 'cnn_model.onnx')
        if os.path.exists(model_file):
            from core.recognizer import MathRecognizer
            print("Loading custom CNN Math Recognizer...")
            self.recognizer = MathRecognizer(model_file)
        else:
            self.recognizer = None

    def paste_to_canvas(self, box, x, y):
        h, w = box.shape[:2]
        ch, cw = self.canvas.shape[:2]
        
        y1, y2 = max(0, y), min(ch, y + h)
        x1, x2 = max(0, x), min(cw, x + w)
        
        box_y1 = y1 - y
        box_y2 = box_y1 + (y2 - y1)
        box_x1 = x1 - x
        box_x2 = box_x1 + (x2 - x1)
        
        if y1 < y2 and x1 < x2:
            roi = self.canvas[y1:y2, x1:x2]
            box_roi = box[box_y1:box_y2, box_x1:box_x2]
            self.canvas[y1:y2, x1:x2] = cv2.bitwise_or(roi, box_roi)

    def process_frame(self, img):
        img = cv2.flip(img, 1)
        
        if self.canvas is None or self.canvas.shape != img.shape:
            self.canvas = np.zeros_like(img)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        save_triggered = False
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        detection_result = self.detector.detect(mp_image)
        
        current_time = time.time()
        eraser_rects = []
        clear_all_triggered = False
        is_thumbs_up_now = False
        thumb_pos_for_draw = None
        
        if detection_result.hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
                handedness = detection_result.handedness[hand_idx][0].category_name
                
                h, w, c = img.shape
                lm_list = []
                for id, lm in enumerate(hand_landmarks):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append([id, cx, cy])
                
                if self.debug and len(lm_list) > 0:
                    for connection in HAND_CONNECTIONS:
                        idx1, idx2 = connection
                        cv2.line(img, (lm_list[idx1][1], lm_list[idx1][2]), 
                                      (lm_list[idx2][1], lm_list[idx2][2]), (255, 0, 0), 2)
                    for lm in lm_list:
                        cv2.circle(img, (lm[1], lm[2]), 4, (0, 0, 255), cv2.FILLED)
                
                if len(lm_list) != 0:
                    tip_ids = [8, 12, 16, 20]
                    fingers = []
                    for id in tip_ids:
                        if lm_list[id][2] < lm_list[id - 2][2]: fingers.append(1)
                        else: fingers.append(0)

                    if handedness == "Left":  # Physical Right hand (Drawing / Dragging)
                        x8, y8 = lm_list[8][1], lm_list[8][2]
                        x4, y4 = lm_list[4][1], lm_list[4][2]
                        
                        # Check Thumbs Up to Save: all 4 fingers closed, thumb tip higher than thumb IP joint
                        if fingers == [0, 0, 0, 0] and lm_list[4][2] < lm_list[3][2]:
                            is_thumbs_up_now = True
                            thumb_pos_for_draw = (x4, y4)

                        pinch_dist = math.hypot(x8 - x4, y8 - y4)
                        
                        # Pinch to Drag
                        if pinch_dist < 40:
                            if self.dragged_box is None:
                                for (bx, by, bw, bh) in self.last_bboxes:
                                    if bx - 20 < x8 < bx + bw + 20 and by - 20 < y8 < by + bh + 20:
                                        self.dragged_box = self.canvas[by:by+bh, bx:bx+bw].copy()
                                        self.canvas[by:by+bh, bx:bx+bw] = 0
                                        self.drag_offset_x = x8 - bx
                                        self.drag_offset_y = y8 - by
                                        self.drag_px, self.drag_py = bx, by
                                        break
                                        
                            if self.dragged_box is not None:
                                self.drag_px = x8 - self.drag_offset_x
                                self.drag_py = y8 - self.drag_offset_y
                                cv2.circle(img, (x8, y8), 15, (255, 0, 255), cv2.FILLED)
                                
                        else:
                            if self.dragged_box is not None:
                                self.paste_to_canvas(self.dragged_box, self.drag_px, self.drag_py)
                                self.dragged_box = None
                                
                            tip_ids = [8, 12, 16, 20]
                            fingers = []
                            for id in tip_ids:
                                if lm_list[id][2] < lm_list[id - 2][2]: fingers.append(1)
                                else: fingers.append(0)
                                
                            if fingers[0] == 1 and fingers[1] == 0:
                                if self.writing_start_time is None:
                                    self.writing_start_time = current_time
                                    
                                if current_time - self.writing_start_time > 0.5:
                                    cv2.circle(img, (x8, y8), 15, self.draw_color, cv2.FILLED)
                                    self.current_stroke.append((x8, y8))
                                else:
                                    cv2.circle(img, (x8, y8), 15, (0, 255, 255), 2)
                            else:
                                self.writing_start_time = None
                                if len(self.current_stroke) > 0:
                                    if len(self.current_stroke) == 1:
                                        cv2.circle(self.canvas, self.current_stroke[0], self.draw_thickness // 2, self.draw_color, cv2.FILLED)
                                    else:
                                        smoothed = chaikin_smooth(self.current_stroke, iterations=3)
                                        for i in range(len(smoothed) - 1):
                                            cv2.line(self.canvas, smoothed[i], smoothed[i+1], self.draw_color, self.draw_thickness)
                                    self.current_stroke = []
                                
                    elif handedness == "Right":  # Physical Left hand (Eraser / Swipe)
                        x0, y0 = lm_list[0][1], lm_list[0][2]
                        x8, y8 = lm_list[8][1], lm_list[8][2]
                        
                        tip_ids = [8, 12, 16, 20]
                        fingers_up_list = []
                        for id in tip_ids:
                            if lm_list[id][2] < lm_list[id - 2][2]: fingers_up_list.append(1)
                            else: fingers_up_list.append(0)
                                
                        # Swipe Clear All (3 or more fingers up)
                        if sum(fingers_up_list) >= 3:
                            self.left_x_history.append(x0)
                            if len(self.left_x_history) == 10:
                                if max(self.left_x_history) - min(self.left_x_history) > 200:
                                    self.canvas = np.zeros_like(img)
                                    self.left_x_history.clear()
                                    clear_all_triggered = True
                        else:
                            self.left_x_history.clear()
                            
                        # Eraser Brush
                        if fingers_up_list[0] == 1 and fingers_up_list[1] == 0:
                            cv2.circle(img, (x8, y8), 15, (0, 0, 255), cv2.FILLED)
                            if self.exp == 0 and self.eyp == 0:
                                self.exp, self.eyp = x8, y8
                            cv2.line(self.canvas, (self.exp, self.eyp), (x8, y8), (0, 0, 0), self.eraser_thickness)
                            self.exp, self.eyp = x8, y8
                        else:
                            self.exp, self.eyp = 0, 0
                            
                        # Group Eraser (Closed Fist)
                        if sum(fingers_up_list) == 0:
                            x_coords = [lm[1] for lm in lm_list]
                            y_coords = [lm[2] for lm in lm_list]
                            ex1, ey1 = max(0, min(x_coords) - 20), max(0, min(y_coords) - 20)
                            ex2, ey2 = min(w, max(x_coords) + 20), min(h, max(y_coords) + 20)
                            eraser_rects.append((ex1, ey1, ex2, ey2))
                            cv2.rectangle(img, (ex1, ey1), (ex2, ey2), (0, 0, 255), 2)
                            
        # No CLEARED! text overlay needed for presentation mode

        if is_thumbs_up_now:
            self.thumbs_up_last_seen_time = current_time
            if self.thumbs_up_start_time is None:
                self.thumbs_up_start_time = current_time
            self.last_thumb_pos = thumb_pos_for_draw
        else:
            if current_time - getattr(self, 'thumbs_up_last_seen_time', 0) > 0.3:
                self.thumbs_up_start_time = None
                
        if self.thumbs_up_start_time is not None:
            time_since_last_save = current_time - getattr(self, 'last_saved_time', 0)
            if time_since_last_save >= 5.0:
                elapsed = current_time - self.thumbs_up_start_time
                if elapsed >= 1.0:
                    save_triggered = True
                    self.last_saved_time = current_time
                    self.thumbs_up_start_time = None
                else:
                    if self.last_thumb_pos is not None:
                        cx, cy = self.last_thumb_pos[0], self.last_thumb_pos[1] - 50
                        cv2.line(img, (cx - 50, cy), (cx + 50, cy), (50, 50, 50), 6)
                        length = int(100 * (1.0 - elapsed))
                        if length > 0:
                            cv2.line(img, (cx - 50, cy), (cx - 50 + length, cy), (0, 255, 255), 6)

        if current_time - getattr(self, 'last_saved_time', 0) < 1.0:
            pass # Removed from OpenCV frame, now handled purely by React UI

        # -------------------------------------------------------------------
        # Grouping, Erasing, CNN Inference
        # -------------------------------------------------------------------
        if not hasattr(self, 'math_parser'):
            from core.math_parser import MathParser
            self.math_parser = MathParser(self.recognizer)
            
        img, equation_str, res_str, self.last_bboxes = self.math_parser.parse_and_draw(img, self.canvas, eraser_rects, self.debug)

        # Merge canvas
        img_gray_updated = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray_updated, 50, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, img_inv)
        img = cv2.bitwise_or(img, self.canvas)

        # Overlay the active smoothed stroke being drawn
        if len(self.current_stroke) > 0:
            if len(self.current_stroke) == 1:
                cv2.circle(img, self.current_stroke[0], self.draw_thickness // 2, self.draw_color, cv2.FILLED)
            else:
                smoothed = chaikin_smooth(self.current_stroke, iterations=3)
                for i in range(len(smoothed) - 1):
                    cv2.line(img, smoothed[i], smoothed[i+1], self.draw_color, self.draw_thickness)

        # Draw the dragged box onto `img` temporarily so it floats
        if self.dragged_box is not None:
            dh, dw = self.dragged_box.shape[:2]
            dx, dy = self.drag_px, self.drag_py
            
            y1, y2 = max(0, dy), min(img.shape[0], dy + dh)
            x1, x2 = max(0, dx), min(img.shape[1], dx + dw)
            
            box_y1 = y1 - dy
            box_y2 = box_y1 + (y2 - y1)
            box_x1 = x1 - dx
            box_x2 = box_x1 + (x2 - x1)
            
            if y1 < y2 and x1 < x2:
                drag_roi = self.dragged_box[box_y1:box_y2, box_x1:box_x2]
                drag_gray = cv2.cvtColor(drag_roi, cv2.COLOR_BGR2GRAY)
                _, drag_mask = cv2.threshold(drag_gray, 50, 255, cv2.THRESH_BINARY)
                drag_mask_inv = cv2.bitwise_not(drag_mask)
                
                img_roi = img[y1:y2, x1:x2]
                img_bg = cv2.bitwise_and(img_roi, img_roi, mask=drag_mask_inv)
                drag_fg = cv2.bitwise_and(drag_roi, drag_roi, mask=drag_mask)
                
                img[y1:y2, x1:x2] = cv2.add(img_bg, drag_fg)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

        return img, equation_str, res_str, save_triggered
