import cv2
import numpy as np

class MathParser:
    def __init__(self, recognizer=None):
        self.recognizer = recognizer
        self.prediction_cache = {}

    def parse_and_draw(self, img, canvas, eraser_rects, debug_mode=False):
        img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
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
                    cv2.drawContours(canvas, [cnt], -1, (0, 0, 0), cv2.FILLED)
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
                            else: display_label = label
                            
                            predictions.append((x, display_label, cx1, cy1))
                                
                            if display_label in ['+', '-', '*', '/']: box_color = (0, 165, 255)
                            elif display_label == "?": box_color = (0, 0, 255)
                            else: box_color = (255, 255, 0)
                                
                            if debug_mode:
                                cv2.rectangle(img, (x, y), (x+bw, y+bh), box_color, 2)
                                cv2.putText(img, display_label.replace('*', 'x'), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 2)
                                cv2.putText(img, f"{confidence*100:.0f}%", (x, y + bh + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
                    else:
                        if debug_mode:
                            cv2.rectangle(img, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
                        
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

        return img, equation_str, res_str, current_bboxes
