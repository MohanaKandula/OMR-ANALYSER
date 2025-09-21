import os
import sys
import logging

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    # Import the enhanced Flask app
    from web_app.app import app
    print("✅ Enhanced OMR Flask app imported successfully")
except ImportError as e:
    print(f"❌ Failed to import Flask app: {e}")
    # Create a minimal fallback app
    from flask import Flask, jsonify
    
    app = Flask(__name__)
    
    @app.route('/')
    def fallback():
        return jsonify({
            'error': 'OMR system failed to initialize',
            'details': str(e),
            'status': 'error'
        }), 500
    
    @app.route('/health')
    def health():
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 503

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8501))
    debug = os.environ.get('ENVIRONMENT') != 'production'
    
    print(f"🚀 Starting OMR application on port {port}")
    print(f"🔧 Debug mode: {debug}")
    print(f"🌍 Environment: {os.environ.get('ENVIRONMENT', 'development')}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)

