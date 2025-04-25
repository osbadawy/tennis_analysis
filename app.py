from flask import Flask, request, jsonify
import os
import tempfile
from main import main
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import json
import traceback

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'input_videos'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/process_video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Get player heights from request form data
    player_1_height = float(request.form.get('player_1_height', 1.88))
    player_2_height = float(request.form.get('player_2_height', 1.91))
    
    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'input_video.mp4')
        file.save(filepath)
        
        try:
            # Process the video with player heights
            result = main(filepath, player_1_height, player_2_height)
            
            return jsonify({
                'message': 'Video processed successfully',
                'player_stats': result
            }), 200
                
        except Exception as e:
            # Log the full error traceback
            error_traceback = traceback.format_exc()
            print(f"Error processing video: {error_traceback}")
            return jsonify({
                'error': str(e),
                'traceback': error_traceback
            }), 500
        finally:
            # Clean up input file
            if os.path.exists(filepath):
                os.remove(filepath)
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 