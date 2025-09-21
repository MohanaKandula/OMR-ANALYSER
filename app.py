import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print(f"🔍 Looking for Flask app in: {current_dir}")
print(f"📁 Directory contents: {os.listdir(current_dir)}")

try:
    # Try to import from web_app directory
    if os.path.exists(os.path.join(current_dir, 'web_app')):
        print("📂 Found web_app directory")
        from web_app.app import app
        print("✅ Successfully imported Flask app from web_app")
    else:
        print("❌ web_app directory not found")
        raise ImportError("web_app directory not found")
        
except ImportError as e:
    print(f"❌ Failed to import Flask app: {e}")
    print("🔧 Creating minimal fallback Flask app")
    
    from flask import Flask, jsonify, render_template_string
    
    app = Flask(__name__)
    
    @app.route('/')
    def fallback():
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>OMR System - Deployment Issue</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; padding: 2rem; background: #f5f5f5; }
                .container { max-width: 600px; margin: 0 auto; background: white; padding: 2rem; border-radius: 10px; }
                .error { color: #E53E3E; }
                .info { color: #2B6CB0; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎯 OMR Evaluation System</h1>
                <p class="error">⚠️ System is initializing...</p>
                <p class="info">The full application failed to load. Please check:</p>
                <ul style="text-align: left;">
                    <li>All required files are present</li>
                    <li>Dependencies are correctly installed</li>
                    <li>File paths are correct</li>
                </ul>
                <p><strong>Error details:</strong> {{ error }}</p>
                <a href="/health" style="background: #4299E1; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Check Health</a>
            </div>
        </body>
        </html>
        """
        return render_template_string(html_template, error=str(e))
    
    @app.route('/health')
    def health():
        return jsonify({
            'status': 'partial',
            'error': 'Main application failed to load',
            'details': str(e),
            'available_routes': ['/', '/health']
        }), 200

# Configure for deployment
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8501))
    debug = os.environ.get('ENVIRONMENT') != 'production'
    
    print(f"🚀 Starting OMR application on port {port}")
    print(f"🔧 Debug mode: {debug}")
    print(f"🌍 Environment: {os.environ.get('ENVIRONMENT', 'development')}")
    print(f"📍 Current working directory: {os.getcwd()}")
    
    # List all files for debugging
    print("📁 Root directory files:")
    for item in os.listdir('.'):
        if os.path.isdir(item):
            print(f"  📂 {item}/")
        else:
            print(f"  📄 {item}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)