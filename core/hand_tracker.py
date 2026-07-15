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
        
        # Performance Cache
        self.prediction_cache = {}

        # Recognizer
        model_file = os.path.join(os.path.dirname(__file__), '..', 'training', 'models', 'cnn_model.pth')
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
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        detection_result = self.detector.detect(mp_image)
        
        current_time = time.time()
        eraser_rects = []
        clear_all_triggered = False
        
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
                    if handedness == "Left":  # Physical Right hand (Drawing / Dragging)
                        x8, y8 = lm_list[8][1], lm_list[8][2]
                        x4, y4 = lm_list[4][1], lm_list[4][2]
                        
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
                            
        if clear_all_triggered:
            cv2.putText(img, "CLEARED!", (img.shape[1]//2 - 150, img.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

        # -------------------------------------------------------------------
        # Grouping, Erasing, CNN Inference
        # -------------------------------------------------------------------
        img_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY)
        # Halved kernel size for tighter grouping threshold
        kernel = np.ones((20, 20), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        predictions = []
        current_bboxes = []
        new_cache = {}
        
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw > 10 and bh > 10:
                current_bboxes.append((x, y, bw, bh))
                bx1, by1, bx2, by2 = x, y, x + bw, y + bh
                
                erased = False
                for (ex1, ey1, ex2, ey2) in eraser_rects:
                    if not (ex2 < bx1 or ex1 > bx2 or ey2 < by1 or ey1 > by2):
                        erased = True
                        break
                        
                if erased:
                    cv2.drawContours(self.canvas, [cnt], -1, (0, 0, 0), cv2.FILLED)
                else:
                    if self.recognizer:
                        pad = max(bw, bh) // 4
                        cx1, cy1 = max(0, x - pad), max(0, y - pad)
                        cx2, cy2 = min(img.shape[1], x + bw + pad), min(img.shape[0], y + bh + pad)
                        
                        _, inference_thresh = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY)
                        crop = inference_thresh[cy1:cy2, cx1:cx2]
                        if crop.size > 0:
                            crop_bytes = crop.tobytes()
                            if crop_bytes in self.prediction_cache:
                                label, confidence = self.prediction_cache[crop_bytes]
                            else:
                                label, confidence = self.recognizer.predict(crop)
                                
                            new_cache[crop_bytes] = (label, confidence)
                            
                            if confidence < 0.60: display_label = "?"
                            else:
                                display_label = label
                                predictions.append((x, display_label, cx1, cy1))
                                
                            if display_label in ['+', '-', '*', '/']: box_color = (0, 165, 255)
                            elif display_label == "?": box_color = (0, 0, 255)
                            else: box_color = (255, 255, 0)
                                
                            cv2.rectangle(img, (x, y), (x+bw, y+bh), box_color, 2)
                            cv2.putText(img, display_label.replace('*', 'x'), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 2)
                            cv2.putText(img, f"{confidence*100:.0f}%", (x, y + bh + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
                    else:
                        cv2.rectangle(img, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
                        
        self.last_bboxes = current_bboxes
        self.prediction_cache = new_cache
        
        predictions.sort(key=lambda item: item[0])
        equation_str = "".join([p[1] for p in predictions])
        
        res_str = ""
        if len(equation_str) > 0 and self.recognizer:
            import sympy
            try:
                if "?" not in equation_str and not equation_str[-1] in "+-*/":
                    result = sympy.sympify(equation_str)
                    if result.is_real and result.is_finite:
                        val = float(result.evalf())
                        if val.is_integer(): res_str = f"= {int(val)}"
                        else: res_str = f"= {val:.2f}"
            except Exception:
                pass 

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

        return img, equation_str, res_str
