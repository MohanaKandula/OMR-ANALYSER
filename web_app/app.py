# Enhanced Flask Web Application with Camera Support
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
import traceback

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the Enhanced OMR processor
try:
    from omr_engine.enhanced_omr_core import EnhancedOMRProcessor
    print("✅ Enhanced OMR processor imported successfully")
except ImportError as e:
    print(f"❌ Failed to import Enhanced OMR processor: {e}")
    try:
        from omr_engine.omr_core import OMRProcessor as EnhancedOMRProcessor
        print("⚠️ Using fallback OMR processor")
    except ImportError:
        print("❌ No OMR processor available")
        EnhancedOMRProcessor = None

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask app configuration
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

# Environment-based configuration
if os.environ.get('ENVIRONMENT') == 'production':
    # Production settings for Render
    UPLOAD_FOLDER = '/tmp/uploads'
    DATABASE_PATH = '/tmp/results.db'
    app.config['DEBUG'] = False
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'config.json')
    KEY_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'answer_key.json')
else:
    # Development settings
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    UPLOAD_FOLDER = os.path.join(project_root, 'data', 'uploads')
    DATABASE_PATH = os.path.join(project_root, 'data', 'results.db')
    app.config['DEBUG'] = True
    CONFIG_PATH = os.path.join(project_root, 'data', 'config.json')
    KEY_PATH = os.path.join(project_root, 'data', 'answer_key.json')

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.secret_key = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')

# Load configuration and initialize OMR processor
try:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            CONFIG = json.load(f)
        SETS = CONFIG.get('sets', ['A', 'B', 'C', 'D'])
        logger.info(f"✅ Config loaded: {len(SETS)} sets available")
    else:
        logger.warning(f"⚠️ Config file not found at {CONFIG_PATH}")
        CONFIG = {'sets': ['A', 'B', 'C', 'D']}
        SETS = ['A', 'B', 'C', 'D']
    
    if EnhancedOMRProcessor and os.path.exists(KEY_PATH):
        omr_processor = EnhancedOMRProcessor(KEY_PATH, CONFIG_PATH)
        logger.info("✅ Enhanced OMR processor initialized successfully")
    else:
        logger.error(f"❌ Key file not found at {KEY_PATH} or processor not available")
        omr_processor = None
        
except Exception as e:
    logger.error(f"❌ Failed to initialize OMR processor: {e}")
    logger.error(traceback.format_exc())
    SETS = ['A', 'B', 'C', 'D']
    omr_processor = None

# Database initialization and functions
def init_database():
    """Initialize SQLite database with enhanced schema"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Main results table
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
            confidence_avg REAL,
            image_source TEXT DEFAULT 'upload',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            error_message TEXT,
            image_path TEXT,
            processing_info TEXT
        )
    ''')
    
    # Detailed answers table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detailed_answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            result_id TEXT NOT NULL,
            question_number INTEGER NOT NULL,
            subject TEXT NOT NULL,
            marked_answer TEXT,
            correct_answer TEXT,
            is_correct BOOLEAN,
            confidence REAL DEFAULT 0.0,
            is_multi_mark BOOLEAN DEFAULT FALSE,
            is_blank BOOLEAN DEFAULT FALSE,
            FOREIGN KEY (result_id) REFERENCES results (id)
        )
    ''')
    
    # System logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            level TEXT NOT NULL,
            message TEXT NOT NULL,
            details TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("✅ Database initialized successfully")

def save_result_to_db(result_data):
    """Save processing result to database with enhanced data"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    try:
        result_id = str(uuid.uuid4())
        
        # Calculate metrics
        total_questions = len(result_data.get('detailed_answers', []))
        total_score = result_data.get('total_score', 0)
        accuracy_percentage = (total_score / max(total_questions, 1)) * 100 if total_questions > 0 else 0
        
        # Calculate average confidence
        confidences = [ans.get('confidence', 0) for ans in result_data.get('detailed_answers', [])]
        confidence_avg = sum(confidences) / len(confidences) if confidences else 0
        
        # Save main result
        cursor.execute('''
            INSERT INTO results 
            (id, filename, set_id, total_score, total_questions, accuracy_percentage, 
             multi_marks, blank_answers, subject_scores, processing_time, confidence_avg,
             image_source, error_message, image_path, processing_info)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            round(confidence_avg, 3),
            result_data.get('image_source', 'upload'),
            result_data.get('error'),
            result_data.get('image_path', ''),
            json.dumps(result_data.get('processing_info', {}))
        ))
        
        # Save detailed answers
        if 'detailed_answers' in result_data:
            for answer in result_data['detailed_answers']:
                cursor.execute('''
                    INSERT INTO detailed_answers
                    (result_id, question_number, subject, marked_answer, correct_answer,
                     is_correct, confidence, is_multi_mark, is_blank)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result_id,
                    answer['question'],
                    answer['subject'],
                    answer.get('marked'),
                    answer.get('correct'),
                    answer.get('is_correct', False),
                    answer.get('confidence', 0.0),
                    answer.get('is_multi_mark', False),
                    answer.get('is_blank', False)
                ))
        
        conn.commit()
        logger.info(f"✅ Result saved to database: {result_id}")
        return result_id
        
    except Exception as e:
        logger.error(f"❌ Database error: {e}")
        logger.error(traceback.format_exc())
        conn.rollback()
        return None
    finally:
        conn.close()

def get_recent_results(limit=50):
    """Get recent results from database with enhanced data"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            SELECT id, filename, set_id, total_score, total_questions, accuracy_percentage,
                   multi_marks, blank_answers, subject_scores, processing_time, confidence_avg,
                   image_source, created_at, error_message
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
                'processing_time': row[9],
                'confidence_avg': row[10],
                'image_source': row[11],
                'created_at': row[12],
                'error': row[13]
            }
            results.append(result)
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Error fetching results: {e}")
        return []
    finally:
        conn.close()

def log_system_event(level, message, details=None):
    """Log system events to database"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO system_logs (level, message, details)
            VALUES (?, ?, ?)
        ''', (level, message, json.dumps(details) if details else None))
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to log system event: {e}")
    finally:
        conn.close()

# Initialize database
init_database()

# Routes
@app.route('/', methods=['GET'])
def index():
    """Enhanced main page with better error handling"""
    try:
        results = get_recent_results(20)
        
        # Check system status
        system_status = {
            'omr_processor': omr_processor is not None,
            'database': True,
            'total_results': len(results),
            'upload_folder': os.path.exists(UPLOAD_FOLDER),
            'config_loaded': len(CONFIG) > 0
        }
        
        # Try to render enhanced template
        template_path = os.path.join(template_dir, 'enhanced_index.html')
        if os.path.exists(template_path):
            return render_template('enhanced_index.html', 
                                 results=results, 
                                 sets=SETS, 
                                 system_status=system_status)
        else:
            # Fallback to simple HTML
            return create_enhanced_html_page(results, system_status)
            
    except Exception as e:
        logger.error(f"❌ Error loading index page: {e}")
        logger.error(traceback.format_exc())
        return create_error_page(str(e))

def create_enhanced_html_page(results, system_status):
    """Create enhanced HTML page with better styling and functionality"""
    status_indicator = "✅" if system_status['omr_processor'] else "❌"
    status_text = "System Ready" if system_status['omr_processor'] else "System Error"
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Enhanced OMR Evaluation System</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{ 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #333;
                min-height: 100vh;
            }}
            
            .header {{
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                padding: 1.5rem 0;
                text-align: center;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                position: sticky;
                top: 0;
                z-index: 100;
            }}
            
            .header h1 {{
                color: #4A5568;
                font-size: 2.5rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
            }}
            
            .status-badge {{
                display: inline-block;
                padding: 0.5rem 1rem;
                border-radius: 20px;
                font-weight: 600;
                margin: 0.5rem;
                {'background: #C6F6D5; color: #22543D;' if system_status['omr_processor'] else 'background: #FED7D7; color: #742A2A;'}
            }}
            
            .container {{
                max-width: 1400px;
                margin: 2rem auto;
                padding: 0 1rem;
            }}
            
            .upload-section {{
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                padding: 2rem;
                border-radius: 20px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                margin-bottom: 2rem;
            }}
            
            .section-title {{
                color: #2D3748;
                font-size: 1.8rem;
                font-weight: 600;
                margin-bottom: 1.5rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }}
            
            .tabs {{
                display: flex;
                margin-bottom: 2rem;
                border-radius: 12px;
                background: #F7FAFC;
                padding: 0.25rem;
            }}
            
            .tab {{
                flex: 1;
                padding: 0.75rem 1.5rem;
                text-align: center;
                background: transparent;
                border: none;
                border-radius: 10px;
                cursor: pointer;
                transition: all 0.3s ease;
                font-weight: 500;
            }}
            
            .tab.active {{
                background: #4299E1;
                color: white;
                box-shadow: 0 4px 12px rgba(66, 153, 225, 0.3);
            }}
            
            .tab-content {{
                display: none;
            }}
            
            .tab-content.active {{
                display: block;
            }}
            
            .form-group {{
                margin-bottom: 1.5rem;
            }}
            
            .form-label {{
                display: block;
                margin-bottom: 0.5rem;
                font-weight: 600;
                color: #2D3748;
            }}
            
            .form-input, .form-select {{
                width: 100%;
                padding: 0.75rem;
                border: 2px solid #E2E8F0;
                border-radius: 10px;
                font-size: 1rem;
                transition: border-color 0.3s ease;
            }}
            
            .form-input:focus, .form-select:focus {{
                outline: none;
                border-color: #4299E1;
                box-shadow: 0 0 0 3px rgba(66, 153, 225, 0.1);
            }}
            
            .btn-primary {{
                background: linear-gradient(135deg, #48BB78 0%, #38A169 100%);
                color: white;
                padding: 1rem 2rem;
                border: none;
                border-radius: 12px;
                font-size: 1.1rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                width: 100%;
            }}
            
            .btn-primary:hover {{
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(72, 187, 120, 0.3);
            }}
            
            .btn-primary:disabled {{
                background: #CBD5E0;
                cursor: not-allowed;
                transform: none;
                box-shadow: none;
            }}
            
            .camera-section {{
                text-align: center;
            }}
            
            .camera-preview {{
                position: relative;
                width: 100%;
                max-width: 500px;
                margin: 0 auto 2rem;
                background: #000;
                border-radius: 15px;
                overflow: hidden;
                box-shadow: 0 8px 32px rgba(0,0,0,0.2);
            }}
            
            #video {{
                width: 100%;
                height: auto;
                display: block;
            }}
            
            .camera-controls {{
                display: flex;
                gap: 1rem;
                justify-content: center;
                flex-wrap: wrap;
            }}
            
            .btn-camera {{
                padding: 1rem 2rem;
                border: none;
                border-radius: 10px;
                font-size: 1rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                min-width: 140px;
            }}
            
            .btn-start {{
                background: #48BB78;
                color: white;
            }}
            
            .btn-capture {{
                background: #4299E1;
                color: white;
            }}
            
            .btn-stop {{
                background: #E53E3E;
                color: white;
            }}
            
            .btn-camera:hover {{
                transform: translateY(-2px);
            }}
            
            .btn-camera:disabled {{
                background: #CBD5E0;
                cursor: not-allowed;
                transform: none;
            }}
            
            .results-section {{
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                padding: 2rem;
                border-radius: 20px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            }}
            
            .results-table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 1rem;
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }}
            
            .results-table th {{
                background: linear-gradient(135deg, #4A5568 0%, #2D3748 100%);
                color: white;
                padding: 1rem;
                text-align: left;
                font-weight: 600;
            }}
            
            .results-table td {{
                padding: 1rem;
                border-bottom: 1px solid #E2E8F0;
                background: white;
            }}
            
            .results-table tr:hover td {{
                background: #F7FAFC;
            }}
            
            .status-success {{ color: #38A169; font-weight: 600; }}
            .status-error {{ color: #E53E3E; font-weight: 600; }}
            
            .alert {{
                padding: 1rem;
                border-radius: 10px;
                margin-bottom: 1rem;
                display: none;
            }}
            
            .alert.show {{
                display: block;
            }}
            
            .alert-success {{
                background: #C6F6D5;
                color: #22543D;
                border: 1px solid #9AE6B4;
            }}
            
            .alert-error {{
                background: #FED7D7;
                color: #742A2A;
                border: 1px solid #FEB2B2;
            }}
            
            .loading {{
                display: none;
                text-align: center;
                padding: 2rem;
            }}
            
            .loading.show {{
                display: block;
            }}
            
            .spinner {{
                border: 3px solid #f3f3f3;
                border-top: 3px solid #4299E1;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto 1rem;
            }}
            
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            
            @media (max-width: 768px) {{
                .header h1 {{
                    font-size: 2rem;
                }}
                
                .container {{
                    margin: 1rem auto;
                    padding: 0 0.5rem;
                }}
                
                .camera-controls {{
                    flex-direction: column;
                }}
                
                .results-table {{
                    font-size: 0.9rem;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🎯 Enhanced OMR Evaluation System</h1>
            <div class="status-badge">{status_indicator} {status_text}</div>
        </div>

        <div class="container">
            <!-- Upload Section -->
            <div class="upload-section">
                <h2 class="section-title">
                    📊 Process OMR Sheets
                </h2>

                <!-- Tabs -->
                <div class="tabs">
                    <button class="tab active" onclick="switchTab('upload')">📁 File Upload</button>
                    <button class="tab" onclick="switchTab('camera')">📷 Camera Capture</button>
                </div>

                <!-- Alerts -->
                <div id="alert-success" class="alert alert-success">
                    <strong>Success!</strong> <span id="alert-success-message"></span>
                </div>
                <div id="alert-error" class="alert alert-error">
                    <strong>Error!</strong> <span id="alert-error-message"></span>
                </div>

                <!-- File Upload Tab -->
                <div id="upload-tab" class="tab-content active">
                    <form id="uploadForm" action="/upload" method="post" enctype="multipart/form-data">
                        <div class="form-group">
                            <label class="form-label">Select Answer Sheet Set:</label>
                            <select name="set_id" id="uploadSetId" class="form-select" required>
                                <option value="">Choose a set...</option>
    """
    
    for set_id in SETS:
        html += f'                                <option value="{set_id}">Set {set_id}</option>\n'
    
    html += f"""
                            </select>
                        </div>

                        <div class="form-group">
                            <label class="form-label">Select Image Files:</label>
                            <input type="file" name="files[]" class="form-input" multiple accept="image/*" required>
                        </div>

                        <button type="submit" class="btn-primary" {'disabled' if not system_status['omr_processor'] else ''}>
                            🚀 Process Images
                        </button>
                    </form>
                </div>

                <!-- Camera Capture Tab -->
                <div id="camera-tab" class="tab-content camera-section">
                    <div class="form-group">
                        <label class="form-label">Select Answer Sheet Set:</label>
                        <select id="cameraSetId" class="form-select" required>
                            <option value="">Choose a set...</option>
    """
    
    for set_id in SETS:
        html += f'                            <option value="{set_id}">Set {set_id}</option>\n'
    
    html += f"""
                        </select>
                    </div>

                    <div class="camera-preview">
                        <video id="video" autoplay playsinline style="display: none;"></video>
                        <div id="camera-placeholder" style="padding: 4rem; color: #718096;">
                            📷 Camera will appear here
                        </div>
                    </div>

                    <div class="camera-controls">
                        <button id="startCamera" class="btn-camera btn-start" {'disabled' if not system_status['omr_processor'] else ''}>📹 Start Camera</button>
                        <button id="capturePhoto" class="btn-camera btn-capture" disabled>📸 Capture & Process</button>
                        <button id="stopCamera" class="btn-camera btn-stop" disabled>⏹️ Stop Camera</button>
                    </div>
                </div>

                <!-- Loading Indicator -->
                <div id="loading" class="loading">
                    <div class="spinner"></div>
                    <p>Processing your OMR sheets... Please wait.</p>
                </div>
            </div>

            <!-- Results Section -->
            <div class="results-section">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem; flex-wrap: wrap; gap: 1rem;">
                    <h2 class="section-title">📈 Recent Results ({len(results)} processed)</h2>
                    <a href="/export" style="background: linear-gradient(135deg, #805AD5 0%, #6B46C1 100%); color: white; padding: 0.75rem 1.5rem; text-decoration: none; border-radius: 10px; font-weight: 500;">📊 Export CSV</a>
                </div>

                <div style="overflow-x: auto;">
                    <table class="results-table">
                        <thead>
                            <tr>
                                <th>📄 Filename</th>
                                <th>📋 Set</th>
                                <th>🎯 Score</th>
                                <th>📊 Accuracy</th>
                                <th>📷 Source</th>
                                <th>⏱️ Time</th>
                                <th>📅 Date</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody>
    """
    
    for result in results[:15]:  # Show recent results
        status = "❌ Error" if result.get('error') else "✅ Success"
        accuracy = f"{result.get('accuracy_percentage', 0):.1f}%" if result.get('accuracy_percentage') else "N/A"
        source_icon = "📷" if result.get('image_source') == 'camera' else "📁"
        processing_time = f"{result.get('processing_time', 0):.1f}s" if result.get('processing_time') else "N/A"
        
        html += f"""
                            <tr>
                                <td>{result.get('filename', 'N/A')}</td>
                                <td>{result.get('set_id', 'N/A')}</td>
                                <td>{result.get('total_score', 'N/A')}/{result.get('total_questions', 'N/A')}</td>
                                <td>{accuracy}</td>
                                <td>{source_icon} {result.get('image_source', 'upload').title()}</td>
                                <td>{processing_time}</td>
                                <td>{result.get('created_at', 'N/A')[:16] if result.get('created_at') else 'N/A'}</td>
                                <td class="{'status-success' if not result.get('error') else 'status-error'}">{status}</td>
                            </tr>
        """
    
    html += """
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <script>
            // Global variables
            let videoStream = null;
            let isProcessing = false;

            // Initialize page
            document.addEventListener('DOMContentLoaded', function() {
                setupCamera();
                setupFileUpload();
            });

            // Tab switching
            function switchTab(tabName) {
                document.querySelectorAll('.tab-content').forEach(content => {
                    content.classList.remove('active');
                });
                
                document.getElementById(tabName + '-tab').classList.add('active');
                
                document.querySelectorAll('.tab').forEach(tab => {
                    tab.classList.remove('active');
                });
                event.target.classList.add('active');
            }

            // File upload setup
            function setupFileUpload() {
                const form = document.getElementById('uploadForm');
                
                form.addEventListener('submit', function(e) {
                    e.preventDefault();
                    
                    if (isProcessing) return;
                    
                    const formData = new FormData(this);
                    const fileInput = this.querySelector('input[type="file"]');
                    const setSelect = document.getElementById('uploadSetId');
                    
                    if (fileInput.files.length === 0) {
                        showAlert('error', 'Please select at least one image file.');
                        return;
                    }
                    
                    if (!setSelect.value) {
                        showAlert('error', 'Please select an answer sheet set.');
                        return;
                    }
                    
                    uploadFiles(formData);
                });
            }

            // Camera functionality
            function setupCamera() {
                const video = document.getElementById('video');
                const placeholder = document.getElementById('camera-placeholder');
                const startBtn = document.getElementById('startCamera');
                const captureBtn = document.getElementById('capturePhoto');
                const stopBtn = document.getElementById('stopCamera');

                startBtn.addEventListener('click', startCamera);
                captureBtn.addEventListener('click', capturePhoto);
                stopBtn.addEventListener('click', stopCamera);

                async function startCamera() {
                    try {
                        const constraints = {
                            video: {
                                width: { ideal: 1280 },
                                height: { ideal: 720 },
                                facingMode: 'environment'
                            }
                        };

                        videoStream = await navigator.mediaDevices.getUserMedia(constraints);
                        video.srcObject = videoStream;
                        video.style.display = 'block';
                        placeholder.style.display = 'none';

                        startBtn.disabled = true;
                        captureBtn.disabled = false;
                        stopBtn.disabled = false;

                        showAlert('success', 'Camera started successfully. Position your answer sheet in the frame.');

                    } catch (error) {
                        console.error('Camera error:', error);
                        showAlert('error', 'Could not access camera. Please check permissions.');
                    }
                }

                function stopCamera() {
                    if (videoStream) {
                        videoStream.getTracks().forEach(track => track.stop());
                        videoStream = null;
                    }

                    video.srcObject = null;
                    video.style.display = 'none';
                    placeholder.style.display = 'block';
                    
                    startBtn.disabled = false;
                    captureBtn.disabled = true;
                    stopBtn.disabled = true;

                    showAlert('success', 'Camera stopped.');
                }

                async function capturePhoto() {
                    const setId = document.getElementById('cameraSetId').value;
                    
                    if (!setId) {
                        showAlert('error', 'Please select an answer sheet set first.');
                        return;
                    }

                    if (isProcessing) return;

                    try {
                        const canvas = document.createElement('canvas');
                        canvas.width = video.videoWidth;
                        canvas.height = video.videoHeight;

                        const ctx = canvas.getContext('2d');
                        ctx.drawImage(video, 0, 0);

                        const imageData = canvas.toDataURL('image/jpeg', 0.8);

                        await processCameraImage(imageData, setId);

                    } catch (error) {
                        console.error('Capture error:', error);
                        showAlert('error', 'Failed to capture image. Please try again.');
                    }
                }
            }

            // Upload files
            async function uploadFiles(formData) {
                setProcessing(true);
                hideAlerts();

                try {
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });

                    if (response.ok) {
                        showAlert('success', 'Files processed successfully!');
                        setTimeout(() => location.reload(), 2000);
                    } else {
                        const error = await response.text();
                        showAlert('error', `Upload failed: ${error}`);
                    }

                } catch (error) {
                    console.error('Upload error:', error);
                    showAlert('error', 'Network error. Please check your connection.');
                } finally {
                    setProcessing(false);
                }
            }

            // Process camera image
            async function processCameraImage(imageData, setId) {
                setProcessing(true);
                hideAlerts();

                try {
                    const response = await fetch('/capture', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            image: imageData,
                            set_id: setId
                        })
                    });

                    const result = await response.json();

                    if (result.success) {
                        const accuracy = result.accuracy_percentage || 0;
                        showAlert('success', `Image processed! Score: ${result.total_score}/${result.total_questions} (${accuracy.toFixed(1)}%)`);
                        setTimeout(() => location.reload(), 3000);
                    } else {
                        showAlert('error', result.error || 'Processing failed');
                    }

                } catch (error) {
                    console.error('Processing error:', error);
                    showAlert('error', 'Failed to process image. Please try again.');
                } finally {
                    setProcessing(false);
                }
            }

            // Utility functions
            function setProcessing(processing) {
                isProcessing = processing;
                const loading = document.getElementById('loading');
                
                if (processing) {
                    loading.classList.add('show');
                    document.querySelectorAll('button, input, select').forEach(el => {
                        el.disabled = true;
                    });
                } else {
                    loading.classList.remove('show');
                    document.querySelectorAll('button, input, select').forEach(el => {
                        el.disabled = false;
                    });
                    
                    // Re-set camera button states
                    const startBtn = document.getElementById('startCamera');
                    const captureBtn = document.getElementById('capturePhoto');
                    const stopBtn = document.getElementById('stopCamera');
                    
                    if (!videoStream) {
                        startBtn.disabled = false;
                        captureBtn.disabled = true;
                        stopBtn.disabled = true;
                    } else {
                        startBtn.disabled = true;
                        captureBtn.disabled = false;
                        stopBtn.disabled = false;
                    }
                }
            }

            function showAlert(type, message) {
                hideAlerts();
                
                const alert = document.getElementById(`alert-${type}`);
                const messageEl = document.getElementById(`alert-${type}-message`);
                
                messageEl.textContent = message;
                alert.classList.add('show');
                
                setTimeout(() => {
                    alert.classList.remove('show');
                }, 5000);
            }

            function hideAlerts() {
                document.querySelectorAll('.alert').forEach(alert => {
                    alert.classList.remove('show');
                });
            }

            // Handle page unload to stop camera
            window.addEventListener('beforeunload', function() {
                if (videoStream) {
                    videoStream.getTracks().forEach(track => track.stop());
                }
            });
        </script>
    </body>
    </html>
    """
    return html

def create_error_page(error_message):
    """Create error page when system fails to load"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>OMR System Error</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 2rem; text-align: center; background: #f5f5f5; }}
            .error-container {{ max-width: 600px; margin: 0 auto; background: white; padding: 2rem; border-radius: 10px; }}
            .error-icon {{ font-size: 4rem; color: #E53E3E; }}
            .error-message {{ color: #742A2A; margin: 1rem 0; }}
            .retry-btn {{ background: #4299E1; color: white; padding: 1rem 2rem; border: none; border-radius: 8px; cursor: pointer; }}
        </style>
    </head>
    <body>
        <div class="error-container">
            <div class="error-icon">⚠️</div>
            <h1>System Error</h1>
            <p class="error-message">Error: {error_message}</p>
            <button class="retry-btn" onclick="location.reload()">Retry</button>
        </div>
    </body>
    </html>
    """

@app.route('/upload', methods=['POST'])
def upload_and_process():
    """Enhanced file upload with better error handling"""
    if not omr_processor:
        return jsonify({'error': 'OMR processor not available'}), 500
    
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files[]')
    set_id = request.form.get('set_id', 'A')
    
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No files selected'}), 400
    
    processed_count = 0
    errors = []
    
    log_system_event('INFO', f'Processing {len(files)} files for set {set_id}')
    
    for file in files:
        if file.filename == '':
            continue
            
        try:
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_filename = f"{timestamp}_{filename}"
            saved_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            # Save uploaded file
            file.save(saved_path)
            logger.info(f"📁 File saved: {saved_path}")
            
            # Process with timing
            start_time = datetime.now()
            result = omr_processor.process_omr_sheet(saved_path, set_id, is_base64=False)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare result data
            result_data = {
                'filename': filename,
                'set_id': set_id,
                'image_path': saved_path,
                'image_source': 'upload',
                'processing_time': processing_time,
                **result
            }
            
            # Save to database
            result_id = save_result_to_db(result_data)
            if result_id:
                processed_count += 1
                logger.info(f"✅ Processed successfully: {filename}")
            else:
                errors.append(f"Database error for {filename}")
            
        except Exception as e:
            error_msg = f"Error processing {filename}: {str(e)}"
            logger.error(f"❌ {error_msg}")
            logger.error(traceback.format_exc())
            errors.append(error_msg)
            
            # Save error to database
            error_data = {
                'filename': filename,
                'set_id': set_id,
                'error': str(e),
                'processing_time': 0,
                'image_source': 'upload'
            }
            save_result_to_db(error_data)
    
    log_system_event('INFO', f'Upload batch completed: {processed_count} successful, {len(errors)} errors')
    
    if errors and processed_count == 0:
        return jsonify({'error': 'All files failed to process', 'details': errors}), 400
    
    return redirect(url_for('index'))

@app.route('/capture', methods=['POST'])
def capture_and_process():
    """Enhanced camera capture processing"""
    if not omr_processor:
        return jsonify({'success': False, 'error': 'OMR processor not available'}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'image' not in data or 'set_id' not in data:
            return jsonify({'success': False, 'error': 'Missing image data or set_id'}), 400
        
        image_data = data['image']
        set_id = data['set_id']
        
        logger.info(f"📷 Processing camera capture for set {set_id}")
        
        # Generate unique filename for camera capture
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"camera_capture_{timestamp}.jpg"
        
        # Process image with timing
        start_time = datetime.now()
        result = omr_processor.process_omr_sheet(image_data, set_id, is_base64=True)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Prepare result data
        result_data = {
            'filename': filename,
            'set_id': set_id,
            'image_source': 'camera',
            'processing_time': processing_time,
            **result
        }
        
        if 'error' in result:
            logger.error(f"❌ Camera processing failed: {result['error']}")
            save_result_to_db(result_data)
            return jsonify({'success': False, 'error': result['error']})
        
        # Save to database
        result_id = save_result_to_db(result_data)
        if not result_id:
            return jsonify({'success': False, 'error': 'Database error'})
        
        logger.info(f"✅ Camera capture processed successfully")
        log_system_event('INFO', f'Camera capture processed: {result.get("total_score", 0)}/{result.get("total_questions", 0)}')
        
        return jsonify({
            'success': True,
            'result_id': result_id,
            'total_score': result.get('total_score', 0),
            'total_questions': result.get('total_questions', 0),
            'accuracy_percentage': result.get('accuracy_percentage', 0),
            'processing_time': processing_time
        })
        
    except Exception as e:
        logger.error(f"❌ Camera capture error: {e}")
        logger.error(traceback.format_exc())
        
        # Save error to database
        error_data = {
            'filename': f"camera_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
            'set_id': data.get('set_id', 'unknown') if 'data' in locals() else 'unknown',
            'error': str(e),
            'processing_time': 0,
            'image_source': 'camera'
        }
        save_result_to_db(error_data)
        
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/export')
def export_results():
    """Export results to CSV with enhanced data"""
    try:
        results = get_recent_results(1000)
        
        si = StringIO()
        writer = csv.writer(si)
        
        # Enhanced headers
        headers = [
            'Filename', 'Set ID', 'Total Score', 'Total Questions', 'Accuracy %', 
            'Multi Marks', 'Blank Answers', 'Confidence Avg', 'Processing Time (s)',
            'Image Source', 'Created At', 'Error'
        ]
        writer.writerow(headers)
        
        # Write data
        for result in results:
            writer.writerow([
                result['filename'],
                result['set_id'],
                result['total_score'],
                result['total_questions'],
                result['accuracy_percentage'],
                result['multi_marks'],
                result['blank_answers'],
                result['confidence_avg'],
                result['processing_time'],
                result['image_source'],
                result['created_at'],
                result.get('error', '')
            ])
        
        output = si.getvalue()
        si.close()
        
        return Response(
            output, 
            mimetype='text/csv',
            headers={"Content-Disposition": f"attachment; filename=omr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"}
        )
        
    except Exception as e:
        logger.error(f"❌ Export error: {e}")
        return f"Export failed: {str(e)}", 500

@app.route('/api/stats')
def get_statistics():
    """Get enhanced processing statistics"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    try:
        # Total processed
        cursor.execute('SELECT COUNT(*) FROM results')
        total_processed = cursor.fetchone()[0]
        
        # Success rate
        cursor.execute('SELECT COUNT(*) FROM results WHERE error_message IS NULL')
        successful = cursor.fetchone()[0]
        success_rate = round((successful / max(total_processed, 1)) * 100, 2)
        
        # Average accuracy
        cursor.execute('SELECT AVG(accuracy_percentage) FROM results WHERE accuracy_percentage IS NOT NULL')
        avg_accuracy = cursor.fetchone()[0] or 0
        
        # Recent activity (last 7 days)
        cursor.execute('''
            SELECT COUNT(*) FROM results 
            WHERE created_at > datetime('now', '-7 days')
        ''')
        recent_activity = cursor.fetchone()[0]
        
        # Average processing time
        cursor.execute('SELECT AVG(processing_time) FROM results WHERE processing_time IS NOT NULL')
        avg_processing_time = cursor.fetchone()[0] or 0
        
        # Camera vs upload breakdown
        cursor.execute('SELECT image_source, COUNT(*) FROM results GROUP BY image_source')
        source_breakdown = dict(cursor.fetchall())
        
        conn.close()
        
        return jsonify({
            'total_processed': total_processed,
            'success_rate': success_rate,
            'average_accuracy': round(avg_accuracy, 2),
            'recent_activity': recent_activity,
            'average_processing_time': round(avg_processing_time, 2),
            'source_breakdown': source_breakdown,
            'system_status': {
                'omr_processor': omr_processor is not None,
                'database': True,
                'upload_folder': os.path.exists(UPLOAD_FOLDER)
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Stats error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/results/<result_id>')
def get_result_details(result_id):
    """Get detailed results for a specific processing"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    try:
        # Get main result
        cursor.execute('''
            SELECT * FROM results WHERE id = ?
        ''', (result_id,))
        
        result_row = cursor.fetchone()
        if not result_row:
            return jsonify({'error': 'Result not found'}), 404
        
        # Get detailed answers
        cursor.execute('''
            SELECT * FROM detailed_answers WHERE result_id = ?
            ORDER BY question_number
        ''', (result_id,))
        
        detailed_answers = cursor.fetchall()
        
        # Format response
        result = {
            'id': result_row[0],
            'filename': result_row[1],
            'set_id': result_row[2],
            'total_score': result_row[3],
            'total_questions': result_row[4],
            'accuracy_percentage': result_row[5],
            'detailed_answers': [
                {
                    'question': ans[2],
                    'subject': ans[3],
                    'marked': ans[4],
                    'correct': ans[5],
                    'is_correct': bool(ans[6]),
                    'confidence': ans[7]
                } for ans in detailed_answers
            ]
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Error getting result details: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/clear_results', methods=['POST'])
def clear_results():
    """Clear all results from database"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM detailed_answers')
        cursor.execute('DELETE FROM results')
        cursor.execute('DELETE FROM system_logs')
        
        conn.commit()
        conn.close()
        
        # Also clear upload folder
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                logger.error(f"Error deleting file {file_path}: {e}")
        
        logger.info("✅ All results cleared successfully")
        return jsonify({'success': True})
        
    except Exception as e:
        logger.error(f"❌ Error clearing results: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'components': {
            'omr_processor': omr_processor is not None,
            'database': True,
            'upload_folder': os.path.exists(UPLOAD_FOLDER),
            'config_loaded': len(CONFIG) > 0
        }
    }
    
    # Check database connectivity
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM results')
        status['components']['database'] = True
        conn.close()
    except Exception:
        status['components']['database'] = False
        status['status'] = 'unhealthy'
    
    # Overall health
    if not all(status['components'].values()):
        status['status'] = 'unhealthy'
    
    status_code = 200 if status['status'] == 'healthy' else 503
    return jsonify(status), status_code

# Error handlers
@app.errorhandler(413)
def file_too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB per file.'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"❌ Internal server error: {e}")
    logger.error(traceback.format_exc())
    return jsonify({'error': 'Internal server error. Please try again.'}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"❌ Unhandled exception: {e}")
    logger.error(traceback.format_exc())
    return jsonify({'error': 'An unexpected error occurred.'}), 500

if __name__ == '__main__':
    print("🚀 Starting Enhanced OMR Web Application...")
    print(f"📁 Template directory: {template_dir}")
    print(f"📂 Upload folder: {UPLOAD_FOLDER}")
    print(f"🗄️ Database: {DATABASE_PATH}")
    print(f"📋 Available sets: {SETS}")
    print(f"🔧 OMR Processor: {'✅ Available' if omr_processor else '❌ Not Available'}")
    print(f"⚙️ Environment: {'🏭 Production' if os.environ.get('ENVIRONMENT') == 'production' else '🔧 Development'}")
    
    port = int(os.environ.get('PORT', 8501))
    debug_mode = os.environ.get('ENVIRONMENT') != 'production'
    
    print(f"🌐 Starting server on port: {port}")
    print("=" * 50)
    
    app.run(
        debug=debug_mode, 
        host='0.0.0.0', 
        port=port, 
        threaded=True
    )