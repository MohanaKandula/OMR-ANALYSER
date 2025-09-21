# Enhanced web_app/app.py - Flask backend with camera support and improved UI
from flask import Flask, request, jsonify, render_template, send_file, redirect, url_for, Response
import os
import base64
from werkzeug.utils import secure_filename
from datetime import datetime
import json
import csv
from io import StringIO, BytesIO
import sys
import sqlite3
import uuid
import logging
from PIL import Image
import cv2
import numpy as np

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the enhanced OMR processor
try:
    from omr_engine.omr_core import EnhancedOMRProcessor as OMRProcessor
except ImportError:
    # Fallback to original if enhanced version not available
    from omr_engine.omr_core import OMRProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = os.path.join(os.getcwd(), '..', 'data', 'uploads')
DATABASE_PATH = os.path.join(os.getcwd(), '..', 'data', 'results.db')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load config and answer key
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'config.json')
KEY_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'answer_key.json')

try:
    with open(CONFIG_PATH) as f:
        CONFIG = json.load(f)
    SETS = CONFIG.get('sets', ['A', 'B'])
    omr = OMRProcessor(KEY_PATH, CONFIG_PATH)
    logger.info("OMR processor initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OMR processor: {e}")
    SETS = ['A', 'B']
    omr = None

# Database initialization
def init_database():
    """Initialize SQLite database for persistent storage"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            set_id TEXT NOT NULL,
            total_score INTEGER,
            total_questions INTEGER,
            accuracy_percentage REAL,
            multi_marks INTEGER DEFAULT 0,
            blank_answers INTEGER DEFAULT 0,
            subject_scores TEXT,
            processing_time REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            error_message TEXT,
            image_path TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detailed_answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            result_id TEXT NOT NULL,
            question_number INTEGER NOT NULL,
            subject TEXT NOT NULL,
            marked_answer TEXT,
            correct_answer TEXT,
            is_correct BOOLEAN,
            is_multi_mark BOOLEAN DEFAULT FALSE,
            is_blank BOOLEAN DEFAULT FALSE,
            FOREIGN KEY (result_id) REFERENCES results (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def save_result_to_db(result_data):
    """Save processing result to database"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    try:
        result_id = str(uuid.uuid4())
        
        # Save main result
        cursor.execute('''
            INSERT INTO results 
            (id, filename, set_id, total_score, total_questions, accuracy_percentage, 
             multi_marks, blank_answers, subject_scores, processing_time, error_message, image_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result_id,
            result_data.get('filename', ''),
            result_data.get('set_id', ''),
            result_data.get('total_score'),
            result_data.get('total_questions'),
            result_data.get('accuracy_percentage'),
            result_data.get('multi_marks', 0),
            result_data.get('blank_answers', 0),
            json.dumps(result_data.get('subject_scores', {})),
            result_data.get('processing_time', 0),
            result_data.get('error'),
            result_data.get('image_path', '')
        ))
        
        # Save detailed answers
        if 'answers' in result_data:
            for answer in result_data['answers']:
                cursor.execute('''
                    INSERT INTO detailed_answers
                    (result_id, question_number, subject, marked_answer, correct_answer,
                     is_correct, is_multi_mark, is_blank)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result_id,
                    answer['question'],
                    answer['subject'],
                    answer['marked'],
                    answer['correct'],
                    answer['is_correct'],
                    answer.get('is_multi_mark', False),
                    answer.get('is_blank', False)
                ))
        
        conn.commit()
        return result_id
        
    except Exception as e:
        logger.error(f"Database error: {e}")
        conn.rollback()
        return None
    finally:
        conn.close()

def get_recent_results(limit=50):
    """Get recent results from database"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, filename, set_id, total_score, total_questions, accuracy_percentage,
               multi_marks, blank_answers, subject_scores, created_at, error_message
        FROM results
        ORDER BY created_at DESC
        LIMIT ?
    ''', (limit,))
    
    results = []
    for row in cursor.fetchall():
        result = {
            'id': row[0],
            'filename': row[1],
            'set_id': row[2],
            'total_score': row[3],
            'total_questions': row[4],
            'accuracy_percentage': row[5],
            'multi_marks': row[6],
            'blank_answers': row[7],
            'subject_scores': json.loads(row[8]) if row[8] else {},
            'created_at': row[9],
            'error': row[10]
        }
        results.append(result)
    
    conn.close()
    return results

# Initialize database
init_database()

@app.route('/', methods=['GET'])
def index():
    """Main page with upload interface and results"""
    try:
        results = get_recent_results()
        return render_template('enhanced_index.html', results=results, sets=SETS)
    except Exception as e:
        logger.error(f"Error loading index page: {e}")
        return render_template('enhanced_index.html', results=[], sets=SETS, error=str(e))

@app.route('/upload', methods=['POST'])
def upload_and_grade():
    """Handle file upload and processing"""
    if 'files[]' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    files = request.files.getlist('files[]')
    set_id = request.form.get('set_id', 'A')
    
    if not omr:
        return jsonify({'error': 'OMR processor not available'}), 500
    
    processed_count = 0
    errors = []
    
    for file in files:
        if file.filename == '':
            continue
            
        try:
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_filename = f"{timestamp}_{filename}"
            saved_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            file.save(saved_path)
            
            # Process with timing
            start_time = datetime.now()
            
            # Use enhanced scoring if available
            if hasattr(omr, 'score_sheet_enhanced'):
                result = omr.score_sheet_enhanced(saved_path, set_id, 
                                                 debug_out=saved_path + '.debug.jpg')
            else:
                result = omr.score_sheet(saved_path, set_id)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare result data
            result_data = {
                'filename': filename,
                'set_id': set_id,
                'image_path': saved_path,
                'processing_time': processing_time,
                **result
            }
            
            # Save to database
            save_result_to_db(result_data)
            processed_count += 1
            
        except Exception as e:
            error_msg = f"Error processing {filename}: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            
            # Save error to database
            error_data = {
                'filename': filename,
                'set_id': set_id,
                'error': str(e),
                'processing_time': 0
            }
            save_result_to_db(error_data)
    
    if errors:
        return jsonify({
            'success': True, 
            'processed': processed_count,
            'errors': errors
        }), 200
    
    return redirect(url_for('index'))

@app.route('/capture', methods=['POST'])
def capture_from_camera():
    """Handle camera capture and processing"""
    try:
        data = request.get_json()
        image_data = data.get('image')
        set_id = data.get('set_id', 'A')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        image_data = image_data.split(',')[1] if ',' in image_data else image_data
        image_bytes = base64.b64decode(image_data)
        
        # Convert to OpenCV format
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Save captured image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"camera_capture_{timestamp}.jpg"
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        cv2.imwrite(saved_path, img)
        
        # Process the image
        start_time = datetime.now()
        
        if hasattr(omr, 'score_sheet_enhanced'):
            result = omr.score_sheet_enhanced(saved_path, set_id, 
                                             debug_out=saved_path + '.debug.jpg')
        else:
            result = omr.score_sheet(saved_path, set_id)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Prepare result data
        result_data = {
            'filename': filename,
            'set_id': set_id,
            'image_path': saved_path,
            'processing_time': processing_time,
            **result
        }
        
        # Save to database
        result_id = save_result_to_db(result_data)
        
        return jsonify({
            'success': True,
            'result_id': result_id,
            'result': result_data
        })
        
    except Exception as e:
        logger.error(f"Camera capture error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/result/<result_id>')
def view_result(result_id):
    """View detailed result"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Get main result
    cursor.execute('''
        SELECT * FROM results WHERE id = ?
    ''', (result_id,))
    
    result_row = cursor.fetchone()
    if not result_row:
        return "Result not found", 404
    
    # Get detailed answers
    cursor.execute('''
        SELECT * FROM detailed_answers WHERE result_id = ?
        ORDER BY question_number
    ''', (result_id,))
    
    answers = cursor.fetchall()
    conn.close()
    
    return render_template('result_detail.html', result=result_row, answers=answers)

@app.route('/export')
def export_results():
    """Export results to CSV"""
    try:
        results = get_recent_results(1000)  # Get more results for export
        
        si = StringIO()
        writer = csv.writer(si)
        
        # Write headers
        headers = [
            'Filename', 'Set ID', 'Total Score', 'Total Questions', 
            'Accuracy %', 'Multi Marks', 'Blank Answers', 'Processing Time (s)',
            'Created At', 'Error'
        ]
        
        # Add subject headers
        if results and results[0]['subject_scores']:
            for subject in results[0]['subject_scores'].keys():
                headers.append(f'{subject} Score')
        
        writer.writerow(headers)
        
        # Write data
        for result in results:
            row = [
                result['filename'],
                result['set_id'],
                result['total_score'],
                result['total_questions'],
                result['accuracy_percentage'],
                result['multi_marks'],
                result['blank_answers'],
                result.get('processing_time', 0),
                result['created_at'],
                result.get('error', '')
            ]
            
            # Add subject scores
            if result['subject_scores']:
                for score in result['subject_scores'].values():
                    row.append(score)
            
            writer.writerow(row)
        
        output = si.getvalue()
        si.close()
        
        return output, 200, {
            'Content-Disposition': 'attachment; filename=omr_results_detailed.csv',
            'Content-Type': 'text/csv'
        }
        
    except Exception as e:
        logger.error(f"Export error: {e}")
        return f"Export failed: {str(e)}", 500

@app.route('/api/stats')
def get_statistics():
    """Get processing statistics"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    try:
        # Overall stats
        cursor.execute('SELECT COUNT(*) FROM results')
        total_processed = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM results WHERE error_message IS NOT NULL')
        total_errors = cursor.fetchone()[0]
        
        cursor.execute('SELECT AVG(accuracy_percentage) FROM results WHERE accuracy_percentage IS NOT NULL')
        avg_accuracy = cursor.fetchone()[0] or 0
        
        cursor.execute('SELECT SUM(multi_marks) FROM results')
        total_multi_marks = cursor.fetchone()[0] or 0
        
        # Recent activity (last 7 days)
        cursor.execute('''
            SELECT COUNT(*) FROM results 
            WHERE created_at >= datetime('now', '-7 days')
        ''')
        recent_activity = cursor.fetchone()[0]
        
        conn.close()
        
        return jsonify({
            'total_processed': total_processed,
            'total_errors': total_errors,
            'success_rate': round((total_processed - total_errors) / max(total_processed, 1) * 100, 2),
            'average_accuracy': round(avg_accuracy, 2),
            'total_multi_marks': total_multi_marks,
            'recent_activity': recent_activity
        })
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/clear_results', methods=['POST'])
def clear_results():
    """Clear all results (admin function)"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM detailed_answers')
        cursor.execute('DELETE FROM results')
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'All results cleared'})
        
    except Exception as e:
        logger.error(f"Clear results error: {e}")
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting Enhanced OMR Web Application...")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Database: {DATABASE_PATH}")
    print(f"Available sets: {SETS}")
    
    # Use PORT environment variable or default to 8501
    port = int(os.environ.get('PORT', 8501))
    print(f"Starting server on port: {port}")
    
    app.run(debug=True, host='0.0.0.0', port=port, threaded=True)