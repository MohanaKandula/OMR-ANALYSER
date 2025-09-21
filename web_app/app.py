# web_app/app.py - Flask backend with UI
from flask import Flask, request, jsonify, render_template, send_file, redirect, url_for
import os
from werkzeug.utils import secure_filename
from omr_engine.omr_core import OMRProcessor
import json
import csv
from io import StringIO
import sys

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

UPLOAD_FOLDER = os.path.join(os.getcwd(), '..', 'data', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Store results in memory
results = []

# Load config and answer key for set options
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'config.json')
KEY_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'answer_key.json')
with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)
SETS = CONFIG.get('sets', ['A'])
omr = OMRProcessor(KEY_PATH, CONFIG_PATH)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', results=results, sets=SETS)

@app.route('/upload', methods=['POST'])
def upload_and_grade():
    if 'files[]' not in request.files:
        return jsonify({'error': 'no file part'}), 400
    files = request.files.getlist('files[]')
    set_id = request.form.get('set_id', 'A')
    for file in files:
        if file.filename == '':
            continue
        filename = secure_filename(file.filename)
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(saved_path)
        try:
            result = omr.score_sheet(saved_path, set_id, debug_out=saved_path + '.warped.jpg')
            result['filename'] = filename
            result['set_id'] = set_id
            results.append(result)
        except Exception as e:
            results.append({
                'filename': filename,
                'set_id': set_id,
                'error': str(e),
                'total_score': 'Error',
                'subject_scores': {}
            })
    return redirect(url_for('index'))

@app.route('/export')
def export_results():
    si = StringIO()
    cw = csv.writer(si)
    # Write headers
    headers = ['filename', 'set_id', 'total_score'] + [s['name'] for s in CONFIG['layout']['subjects']] + ['error']
    cw.writerow(headers)
    # Write rows
    for result in results:
        row = [
            result.get('filename', ''),
            result.get('set_id', ''),
            result.get('total_score', '')
        ]
        for s in CONFIG['layout']['subjects']:
            row.append(result.get('subject_scores', {}).get(s['name'], ''))
        row.append(result.get('error', ''))
        cw.writerow(row)
    output = si.getvalue()
    si.close()
    return output, 200, {
        'Content-Disposition': 'attachment; filename=omr_results.csv',
        'Content-Type': 'text/csv'
    }

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8501)
