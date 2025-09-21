# Fixed web_app/app.py - Flask backend with proper template handling
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

# Import the OMR processor - FIXED IMPORT
try:
    from omr_engine.omr_core import OMRProcessor
    print("OMR processor imported successfully")
except ImportError as e:
    print(f"Failed to import OMR processor: {e}")
    OMRProcessor = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FIXED: Flask app configuration with proper template path
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

# Configuration based on environment
if os.environ.get('ENVIRONMENT') == 'production':
    # Production settings for Render
    UPLOAD_FOLDER = '/tmp/uploads'
    DATABASE_PATH = '/tmp/results.db'
    app.config['DEBUG'] = False
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'config.json')
    KEY_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'answer_key.json')
else:
    # Development settings
    UPLOAD_FOLDER = os.path.join(os.getcwd(), '..', 'data', 'uploads')
    DATABASE_PATH = os.path.join(os.getcwd(), '..', 'data', 'results.db')
    app.config['DEBUG'] = True
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'config.json')
    KEY_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'answer_key.json')

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load config and answer key - FIXED ERROR HANDLING
try:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            CONFIG = json.load(f)
        SETS = CONFIG.get('sets', ['A', 'B', 'C', 'D'])
    else:
        print(f"Config file not found at {CONFIG_PATH}")
        CONFIG = {}
        SETS = ['A', 'B', 'C', 'D']
    
    if OMRProcessor and os.path.exists(KEY_PATH):
        omr = OMRProcessor(KEY_PATH, CONFIG_PATH)
        logger.info("OMR processor initialized successfully")
    else:
        print(f"Key file not found at {KEY_PATH} or OMRProcessor not available")
        omr = None
        
except Exception as e:
    logger.error(f"Failed to initialize OMR processor: {e}")
    SETS = ['A', 'B', 'C', 'D']
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
        
        # Calculate additional metrics
        total_questions = len(result_data.get('answers', []))
        total_score = result_data.get('total_score', 0)
        accuracy_percentage = (total_score / max(total_questions, 1)) * 100 if total_questions > 0 else 0
        
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
            total_score,
            total_questions,
            round(accuracy_percentage, 2),
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

# FIXED: Main route with proper error handling
@app.route('/', methods=['GET'])
def index():
    """Main page with upload interface and results"""
    try:
        results = get_recent_results()
        
        # Check if template exists
        template_path = os.path.join(template_dir, 'enhanced_index.html')
        if os.path.exists(template_path):
            return render_template('enhanced_index.html', results=results, sets=SETS)
        else:
            # Fallback to simple HTML if template not found
            return create_simple_html_page(results)
            
    except Exception as e:
        logger.error(f"Error loading index page: {e}")
        return create_simple_html_page([])

def create_simple_html_page(results):
    """Fallback HTML page if template is missing"""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>OMR Evaluation System</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 2rem; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 2rem; border-radius: 10px; }}
            .header {{ text-align: center; margin-bottom: 2rem; }}
            .upload-form {{ background: #f8f9fa; padding: 2rem; border-radius: 8px; margin-bottom: 2rem; }}
            .form-group {{ margin-bottom: 1rem; }}
            label {{ display: block; margin-bottom: 0.5rem; font-weight: bold; }}
            select, input {{ padding: 0.5rem; border: 1px solid #ddd; border-radius: 4px; width: 100%; max-width: 300px; }}
            button {{ background: #007bff; color: white; padding: 0.75rem 2rem; border: none; border-radius: 4px; cursor: pointer; }}
            button:hover {{ background: #0056b3; }}
            .results {{ margin-top: 2rem; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ padding: 1rem; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background: #f8f9fa; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎯 OMR Evaluation System</h1>
                <p>Advanced bubble sheet processing system</p>
                <p><strong>Status:</strong> ✅ System is running successfully!</p>
            </div>
            
            <div class="upload-form">
                <h2>📊 Process OMR Sheets</h2>
                <form action="/upload" method="post" enctype="multipart/form-data">
                    <div class="form-group">
                        <label>Select Answer Sheet Set:</label>
                        <select name="set_id" required>
                            <option value="">Choose a set...</option>
                            <option value="A">Set A</option>
                            <option value="B">Set B</option>
                            <option value="C">Set C</option>
                            <option value="D">Set D</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label>Select Image Files:</label>
                        <input type="file" name="files[]" multiple accept="image/*" required>
                    </div>
                    
                    <button type="submit">🚀 Process Images</button>
                </form>
            </div>
            
            <div class="results">
                <h2>📈 Recent Results ({len(results)} processed)</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Filename</th>
                            <th>Set</th>
                            <th>Score</th>
                            <th>Accuracy</th>
                            <th>Date</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
    """
    
    for result in results[:10]:  # Show only first 10
        status = "❌ Error" if result.get('error') else "✅ Success"
        accuracy = f"{result.get('accuracy_percentage', 0):.1f}%" if result.get('accuracy_percentage') else "N/A"
        html += f"""
                        <tr>
                            <td>{result.get('filename', 'N/A')}</td>
                            <td>{result.get('set_id', 'N/A')}</td>
                            <td>{result.get('total_score', 'N/A')}/{result.get('total_questions', 'N/A')}</td>
                            <td>{accuracy}</td>
                            <td>{result.get('created_at', 'N/A')}</td>
                            <td>{status}</td>
                        </tr>
        """
    
    html += """
                    </tbody>
                </table>
                <p><a href="/export">📊 Export to CSV</a></p>
            </div>
        </div>
    </body>
    </html>
    """
    return html

@app.route('/upload', methods=['POST'])
def upload_and_grade():
    """Handle file upload and processing"""
    if 'files[]' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    files = request.files.getlist('files[]')
    set_id = request.form.get('set_id', 'A')
    
    if not omr:
        return jsonify({'error': 'OMR processor not available - demo mode'}), 500
    
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
    
    return redirect(url_for('index'))

@app.route('/export')
def export_results():
    """Export results to CSV"""
    try:
        results = get_recent_results(1000)
        
        si = StringIO()
        writer = csv.writer(si)
        
        # Write headers
        headers = ['Filename', 'Set ID', 'Total Score', 'Total Questions', 'Accuracy %', 'Created At', 'Error']
        writer.writerow(headers)
        
        # Write data
        for result in results:
            writer.writerow([
                result['filename'],
                result['set_id'],
                result['total_score'],
                result['total_questions'],
                result['accuracy_percentage'],
                result['created_at'],
                result.get('error', '')
            ])
        
        output = si.getvalue()
        si.close()
        
        return Response(output, mimetype='text/csv', 
                       headers={"Content-Disposition": "attachment; filename=omr_results.csv"})
        
    except Exception as e:
        logger.error(f"Export error: {e}")
        return f"Export failed: {str(e)}", 500

@app.route('/api/stats')
def get_statistics():
    """Get processing statistics"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute('SELECT COUNT(*) FROM results')
        total_processed = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM results WHERE error_message IS NULL')
        successful = cursor.fetchone()[0]
        
        success_rate = round((successful / max(total_processed, 1)) * 100, 2)
        
        cursor.execute('SELECT AVG(accuracy_percentage) FROM results WHERE accuracy_percentage IS NOT NULL')
        avg_accuracy = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return jsonify({
            'total_processed': total_processed,
            'success_rate': success_rate,
            'average_accuracy': round(avg_accuracy, 2),
            'recent_activity': total_processed
        })
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return "Page not found. Try the home page: <a href='/'>Home</a>", 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {e}")
    return f"Internal server error: {str(e)}", 500

if __name__ == '__main__':
    print("Starting OMR Web Application...")
    print(f"Template directory: {template_dir}")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Database: {DATABASE_PATH}")
    print(f"Available sets: {SETS}")
    print(f"OMR Processor: {'✅ Available' if omr else '❌ Not Available'}")
    
    port = int(os.environ.get('PORT', 8501))
    print(f"Starting server on port: {port}")
    
    app.run(debug=True, host='0.0.0.0', port=port, threaded=True)