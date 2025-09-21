

"""
OMR Core Engine
Handles image preprocessing, bubble detection, answer extraction, scoring, and overlays.
Supports multiple sheet versions/sets.
"""
import cv2
import numpy as np
import json
import os
from typing import List, Dict, Tuple, Any

class OMRProcessor:
    def __init__(self, answer_key_path: str, config_path: str):
        self.answer_key = self._load_json(answer_key_path)
        self.config = self._load_json(config_path)

    def _load_json(self, path: str) -> dict:
        with open(path, 'r') as f:
            return json.load(f)

    def load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        return img

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        return blur

    def find_sheet_contour(self, img: np.ndarray) -> Any:
        edged = cv2.Canny(img, 50, 150)
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                return approx
        return None

    def four_point_transform(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        rect = pts.reshape(4, 2)
        s = rect.sum(axis=1)
        diff = np.diff(rect, axis=1)
        tl = rect[np.argmin(s)]
        br = rect[np.argmax(s)]
        tr = rect[np.argmin(diff)]
        bl = rect[np.argmax(diff)]
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))
        dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect.astype('float32'), dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    def extract_bubble_grid(self, warped_image: np.ndarray, rows=20, cols=5) -> List[List[Tuple[int, int, int, int]]]:
        h, w = warped_image.shape[:2]
        cell_h = h // rows
        cell_w = w // cols
        cells = []
        for r in range(rows):
            row_cells = []
            for c in range(cols):
                x1, y1 = c * cell_w, r * cell_h
                x2, y2 = x1 + cell_w, y1 + cell_h
                row_cells.append((x1, y1, x2, y2))
            cells.append(row_cells)
        return cells

    def is_marked(self, cell_img: np.ndarray, thresh=0.6) -> Tuple[bool, float]:
        gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY) if len(cell_img.shape) == 3 else cell_img
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        frac_dark = np.sum(binary == 0) / (binary.size + 1e-6)
        return frac_dark > thresh, frac_dark

    def score_sheet(self, image_path: str, set_id: str, debug_out: str = None) -> Dict[str, Any]:
        config = self.config
        layout = config['layout']
        grid = config['bubble_grid']
        img = self.load_image(image_path)
        pre = self.preprocess(img)
        sheet_cnt = self.find_sheet_contour(pre)
        if sheet_cnt is None:
            raise ValueError("Could not detect sheet contour.")
        warped = self.four_point_transform(img, sheet_cnt)
        warped = cv2.resize(warped, (config['sheet_geometry']['width'], config['sheet_geometry']['height']))
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # Use new answer key structure: set_id -> subject -> list of answers
        key = self.answer_key[set_id]
        subjects = layout['subjects']
        options = ['A', 'B', 'C', 'D']
        answers = []
        subject_scores = {}
        total_correct = 0
        question_idx = 0
        for subj in subjects:
            subj_name = subj['name']
            subj_answers = key[subj_name]
            subj_correct = 0
            for q_subj_idx in range(len(subj_answers)):
                question_no = subj['start'] + q_subj_idx
                # Determine column and row for this question
                col_idx = (question_no - 1) // layout['questions_per_subject']
                q_in_col = (question_no - 1) % layout['questions_per_subject']
                bubbled_option = None
                max_filled = 0
                for opt_idx, option in enumerate(options):
                    b_center_x = grid['column_x_offsets'][col_idx] + (opt_idx * grid['option_x_step'])
                    b_center_y = grid['row_y_start'] + (q_in_col * grid['row_y_step'])
                    mask = np.zeros(thresh.shape, dtype="uint8")
                    cv2.circle(mask, (b_center_x, b_center_y), grid['bubble_radius'], 255, -1)
                    masked = cv2.bitwise_and(thresh, thresh, mask=mask)
                    total = cv2.countNonZero(masked)
                    if total > max_filled:
                        max_filled = total
                        bubbled_option = option
                correct_answer = subj_answers[q_subj_idx]
                is_correct = (bubbled_option == correct_answer)
                if is_correct:
                    subj_correct += 1
                    total_correct += 1
                answers.append({
                    'question': question_no,
                    'subject': subj_name,
                    'marked': bubbled_option,
                    'correct': correct_answer,
                    'is_correct': is_correct
                })
            subject_scores[subj_name] = subj_correct
        results = {
            'answers': answers,
            'subject_scores': subject_scores,
            'total_score': total_correct
        }
        if debug_out:
            debug_img = warped.copy()
            cv2.imwrite(debug_out, debug_img)
        return results

# Example usage:
# omr = OMRProcessor('data/answer_key.json', 'data/config.json')
# result = omr.score_sheet('path/to/image.jpg', set_id='A')

if __name__ == '__main__':
    import argparse, sys
    parser = argparse.ArgumentParser(description='Grade a single OMR sheet (refactored).')
    parser.add_argument('--image', required=True, help='Path to OMR image')
    parser.add_argument('--key', required=True, help='Path to answer key JSON')
    parser.add_argument('--config', required=True, help='Path to config JSON')
    parser.add_argument('--set', required=False, help='Set ID (A/B/C/D)', default=None)
    parser.add_argument('--out', required=False, help='Debug output image path')
    args = parser.parse_args()
    omr = OMRProcessor(args.key, args.config)
    set_id = args.set if args.set else 'A'
    res = omr.score_sheet(args.image, set_id, debug_out=args.out)
    print(json.dumps(res, indent=2))
