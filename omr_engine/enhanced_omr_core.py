"""
Enhanced OMR Core Engine with Advanced Image Processing
Handles mobile camera captures, perspective correction, and accurate bubble detection
"""
import cv2
import numpy as np
import json
import os
import imutils
from typing import List, Dict, Tuple, Any, Optional
import logging
from scipy import ndimage
from skimage import filters, measure, morphology
import base64
from io import BytesIO
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedOMRProcessor:
    """Enhanced OMR processor with advanced computer vision techniques"""
    
    def __init__(self, answer_key_path: str, config_path: str):
        self.answer_key = self._load_json(answer_key_path)
        self.config = self._load_json(config_path)
        self.debug_mode = self.config.get('debug_mode', False)
    
    def _load_json(self, path: str) -> dict:
        """Load JSON configuration files"""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            return {}
    
    def load_image_from_base64(self, base64_string: str) -> np.ndarray:
        """Convert base64 string to OpenCV image"""
        try:
            # Remove data URL prefix if present
            if base64_string.startswith('data:image'):
                base64_string = base64_string.split(',')[1]
            
            # Decode base64
            image_data = base64.b64decode(base64_string)
            
            # Convert to PIL Image then to OpenCV format
            pil_image = Image.open(BytesIO(image_data))
            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            return opencv_image
        except Exception as e:
            logger.error(f"Failed to decode base64 image: {e}")
            raise ValueError(f"Invalid image data: {e}")
    
    def load_image_from_file(self, path: str) -> np.ndarray:
        """Load image from file path"""
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found or corrupted: {path}")
        return img
    
    def enhance_image_quality(self, img: np.ndarray) -> np.ndarray:
        """Enhance image quality for better processing"""
        try:
            # Convert to LAB color space for better processing
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            
            # Split channels
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l_enhanced = clahe.apply(l)
            
            # Merge channels and convert back to BGR
            enhanced_lab = cv2.merge([l_enhanced, a, b])
            enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            # Apply bilateral filter to reduce noise while preserving edges
            enhanced_img = cv2.bilateralFilter(enhanced_img, 9, 75, 75)
            
            return enhanced_img
        except Exception as e:
            logger.warning(f"Image enhancement failed, using original: {e}")
            return img
    
    def detect_sheet_corners(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Detect OMR sheet corners using advanced contour detection"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Find contours
            contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)
            
            # Sort contours by area (largest first)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Find the largest rectangular contour
            for contour in contours:
                # Approximate contour
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                # Check if contour has 4 points (rectangle)
                if len(approx) == 4:
                    # Check if area is reasonable (not too small)
                    area = cv2.contourArea(approx)
                    img_area = img.shape[0] * img.shape[1]
                    
                    if area > 0.1 * img_area:  # At least 10% of image area
                        return approx.reshape(4, 2)
            
            # If no good rectangle found, use whole image
            h, w = img.shape[:2]
            return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Corner detection failed: {e}")
            return None
    
    def apply_perspective_correction(self, img: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Apply perspective correction to get top-down view"""
        try:
            # Order corners: top-left, top-right, bottom-right, bottom-left
            def order_points(pts):
                rect = np.zeros((4, 2), dtype="float32")
                
                # Sum and difference to find corners
                s = pts.sum(axis=1)
                diff = np.diff(pts, axis=1)
                
                rect[0] = pts[np.argmin(s)]      # top-left
                rect[2] = pts[np.argmax(s)]      # bottom-right
                rect[1] = pts[np.argmin(diff)]   # top-right
                rect[3] = pts[np.argmax(diff)]   # bottom-left
                
                return rect
            
            ordered_corners = order_points(corners.astype('float32'))
            
            # Calculate destination dimensions
            width_top = np.linalg.norm(ordered_corners[1] - ordered_corners[0])
            width_bottom = np.linalg.norm(ordered_corners[2] - ordered_corners[3])
            width = int(max(width_top, width_bottom))
            
            height_left = np.linalg.norm(ordered_corners[3] - ordered_corners[0])
            height_right = np.linalg.norm(ordered_corners[2] - ordered_corners[1])
            height = int(max(height_left, height_right))
            
            # Define destination points
            dst = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype="float32")
            
            # Calculate perspective transform matrix
            matrix = cv2.getPerspectiveTransform(ordered_corners, dst)
            
            # Apply perspective correction
            corrected = cv2.warpPerspective(img, matrix, (width, height))
            
            return corrected
            
        except Exception as e:
            logger.error(f"Perspective correction failed: {e}")
            return img
    
    def extract_bubble_regions(self, img: np.ndarray) -> Dict[str, List]:
        """Extract bubble regions based on configuration"""
        try:
            config = self.config
            layout = config['layout']
            grid = config['bubble_grid']
            
            # Resize image to standard dimensions
            target_width = config['sheet_geometry']['width']
            target_height = config['sheet_geometry']['height']
            img_resized = cv2.resize(img, (target_width, target_height))
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            
            bubble_regions = {}
            
            for subject in layout['subjects']:
                subject_name = subject['name']
                start_q = subject['start']
                end_q = subject['end']
                
                bubble_regions[subject_name] = []
                
                for q_num in range(start_q, end_q + 1):
                    question_bubbles = []
                    
                    # Calculate position based on layout
                    col_idx = (q_num - 1) // layout['questions_per_subject']
                    row_idx = (q_num - 1) % layout['questions_per_subject']
                    
                    if col_idx < len(grid['column_x_offsets']):
                        base_x = grid['column_x_offsets'][col_idx]
                        base_y = grid['row_y_start'] + (row_idx * grid['row_y_step'])
                        
                        # Extract each option bubble (A, B, C, D)
                        for opt_idx in range(layout['options_per_question']):
                            bubble_x = base_x + (opt_idx * grid['option_x_step'])
                            bubble_y = base_y
                            radius = grid['bubble_radius']
                            
                            # Extract bubble region
                            x1 = max(0, bubble_x - radius)
                            y1 = max(0, bubble_y - radius)
                            x2 = min(target_width, bubble_x + radius)
                            y2 = min(target_height, bubble_y + radius)
                            
                            bubble_region = gray[y1:y2, x1:x2]
                            question_bubbles.append({
                                'option': chr(65 + opt_idx),  # A, B, C, D
                                'region': bubble_region,
                                'center': (bubble_x, bubble_y),
                                'bounds': (x1, y1, x2, y2)
                            })
                    
                    bubble_regions[subject_name].append({
                        'question': q_num,
                        'bubbles': question_bubbles
                    })
            
            return bubble_regions
            
        except Exception as e:
            logger.error(f"Bubble extraction failed: {e}")
            return {}
    
    def detect_marked_bubbles(self, bubble_regions: Dict[str, List]) -> Dict[str, List]:
        """Detect marked bubbles using advanced techniques"""
        try:
            detected_answers = {}
            
            for subject_name, questions in bubble_regions.items():
                detected_answers[subject_name] = []
                
                for question_data in questions:
                    question_num = question_data['question']
                    bubbles = question_data['bubbles']
                    
                    # Analyze each bubble
                    bubble_scores = []
                    
                    for bubble in bubbles:
                        region = bubble['region']
                        option = bubble['option']
                        
                        if region.size == 0:
                            bubble_scores.append((option, 0.0))
                            continue
                        
                        # Multiple detection techniques
                        score = self._analyze_bubble_region(region)
                        bubble_scores.append((option, score))
                    
                    # Determine marked answer
                    bubble_scores.sort(key=lambda x: x[1], reverse=True)
                    
                    # Check if any bubble is clearly marked
                    if bubble_scores[0][1] > self.config.get('detection_threshold', 0.3):
                        # Check for multiple marks
                        is_multi_mark = (
                            len(bubble_scores) > 1 and 
                            bubble_scores[1][1] > self.config.get('detection_threshold', 0.3)
                        )
                        
                        detected_answers[subject_name].append({
                            'question': question_num,
                            'marked_answer': bubble_scores[0][0],
                            'confidence': bubble_scores[0][1],
                            'is_multi_mark': is_multi_mark,
                            'all_scores': bubble_scores
                        })
                    else:
                        # No clear mark detected
                        detected_answers[subject_name].append({
                            'question': question_num,
                            'marked_answer': None,
                            'confidence': 0.0,
                            'is_multi_mark': False,
                            'all_scores': bubble_scores
                        })
            
            return detected_answers
            
        except Exception as e:
            logger.error(f"Bubble detection failed: {e}")
            return {}
    
    def _analyze_bubble_region(self, region: np.ndarray) -> float:
        """Analyze a single bubble region to determine if it's marked"""
        try:
            if region.size == 0:
                return 0.0
            
            # Method 1: Adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            filled_pixels = np.sum(thresh > 0)
            total_pixels = thresh.size
            fill_ratio = filled_pixels / total_pixels
            
            # Method 2: Edge density analysis
            edges = cv2.Canny(region, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Method 3: Variance analysis (marked bubbles have lower variance)
            variance = np.var(region) / 255.0
            variance_score = max(0, 1 - variance)
            
            # Method 4: Circular mask analysis
            h, w = region.shape
            center = (w // 2, h // 2)
            radius = min(w, h) // 3
            
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, center, radius, 255, -1)
            
            masked_region = cv2.bitwise_and(region, mask)
            mean_intensity = np.mean(masked_region[mask > 0]) if np.sum(mask) > 0 else 255
            intensity_score = max(0, 1 - mean_intensity / 255.0)
            
            # Combine scores with weights
            final_score = (
                fill_ratio * 0.4 +
                intensity_score * 0.3 +
                variance_score * 0.2 +
                edge_density * 0.1
            )
            
            return min(1.0, final_score)
            
        except Exception as e:
            logger.warning(f"Bubble analysis failed: {e}")
            return 0.0
    
    def calculate_scores(self, detected_answers: Dict[str, List], set_id: str) -> Dict[str, Any]:
        """Calculate subject-wise and total scores"""
        try:
            if set_id not in self.answer_key:
                raise ValueError(f"Answer key not found for set {set_id}")
            
            answer_key = self.answer_key[set_id]
            results = {
                'set_id': set_id,
                'subject_scores': {},
                'total_score': 0,
                'total_questions': 0,
                'multi_marks': 0,
                'blank_answers': 0,
                'detailed_answers': []
            }
            
            for subject_name, questions in detected_answers.items():
                if subject_name not in answer_key:
                    continue
                
                subject_key = answer_key[subject_name]
                subject_score = 0
                
                for i, question_data in enumerate(questions):
                    if i >= len(subject_key):
                        continue
                    
                    question_num = question_data['question']
                    marked_answer = question_data['marked_answer']
                    correct_answer = subject_key[i]
                    confidence = question_data['confidence']
                    is_multi_mark = question_data['is_multi_mark']
                    
                    # Handle multiple correct answers (like "A,B,C,D")
                    if isinstance(correct_answer, str) and ',' in correct_answer:
                        correct_answers = [ans.strip() for ans in correct_answer.split(',')]
                        is_correct = marked_answer in correct_answers if marked_answer else False
                    else:
                        is_correct = marked_answer == correct_answer if marked_answer else False
                    
                    # Count statistics
                    if is_correct:
                        subject_score += 1
                    
                    if is_multi_mark:
                        results['multi_marks'] += 1
                    
                    if marked_answer is None:
                        results['blank_answers'] += 1
                    
                    results['detailed_answers'].append({
                        'question': question_num,
                        'subject': subject_name,
                        'marked': marked_answer,
                        'correct': correct_answer,
                        'is_correct': is_correct,
                        'confidence': confidence,
                        'is_multi_mark': is_multi_mark,
                        'is_blank': marked_answer is None
                    })
                
                results['subject_scores'][subject_name] = subject_score
                results['total_score'] += subject_score
                results['total_questions'] += len(questions)
            
            # Calculate accuracy
            results['accuracy_percentage'] = (
                (results['total_score'] / max(results['total_questions'], 1)) * 100
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Score calculation failed: {e}")
            return {'error': str(e)}
    
    def process_omr_sheet(self, image_input, set_id: str, is_base64: bool = False) -> Dict[str, Any]:
        """Main processing pipeline"""
        try:
            # Load image
            if is_base64:
                img = self.load_image_from_base64(image_input)
            else:
                img = self.load_image_from_file(image_input)
            
            logger.info(f"Processing image of size: {img.shape}")
            
            # Step 1: Enhance image quality
            enhanced_img = self.enhance_image_quality(img)
            
            # Step 2: Detect sheet corners
            corners = self.detect_sheet_corners(enhanced_img)
            if corners is None:
                raise ValueError("Could not detect OMR sheet in image")
            
            # Step 3: Apply perspective correction
            corrected_img = self.apply_perspective_correction(enhanced_img, corners)
            
            # Step 4: Extract bubble regions
            bubble_regions = self.extract_bubble_regions(corrected_img)
            if not bubble_regions:
                raise ValueError("Could not extract bubble regions")
            
            # Step 5: Detect marked bubbles
            detected_answers = self.detect_marked_bubbles(bubble_regions)
            
            # Step 6: Calculate scores
            results = self.calculate_scores(detected_answers, set_id)
            
            # Add processing metadata
            results['processing_info'] = {
                'original_size': f"{img.shape[1]}x{img.shape[0]}",
                'corrected_size': f"{corrected_img.shape[1]}x{corrected_img.shape[0]}",
                'corners_detected': corners is not None,
                'enhancement_applied': True
            }
            
            return results
            
        except Exception as e:
            logger.error(f"OMR processing failed: {e}")
            return {
                'error': str(e),
                'processing_info': {
                    'failed_at': 'processing_pipeline',
                    'error_details': str(e)
                }
            }
    
    def create_debug_overlay(self, img: np.ndarray, results: Dict[str, Any], save_path: str):
        """Create debug overlay showing detection results"""
        try:
            overlay = img.copy()
            
            if 'detailed_answers' in results:
                for answer in results['detailed_answers']:
                    question = answer['question']
                    is_correct = answer.get('is_correct', False)
                    confidence = answer.get('confidence', 0)
                    
                    # Draw indicators (simplified for now)
                    color = (0, 255, 0) if is_correct else (0, 0, 255)
                    # Add visualization logic here
            
            cv2.imwrite(save_path, overlay)
            logger.info(f"Debug overlay saved to: {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to create debug overlay: {e}")

# Example usage and testing
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced OMR Processing')
    parser.add_argument('--image', required=True, help='Path to OMR image')
    parser.add_argument('--set', required=True, help='Answer set (A/B/C/D)')
    parser.add_argument('--key', required=True, help='Path to answer key JSON')
    parser.add_argument('--config', required=True, help='Path to config JSON')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = EnhancedOMRProcessor(args.key, args.config)
    
    # Process image
    results = processor.process_omr_sheet(args.image, args.set, is_base64=False)
    
    # Print results
    print(json.dumps(results, indent=2))
    
    # Save debug overlay if requested
    if args.debug and 'error' not in results:
        debug_path = args.image.replace('.jpg', '_debug.jpg').replace('.png', '_debug.png')
        img = processor.load_image_from_file(args.image)
        processor.create_debug_overlay(img, results, debug_path)