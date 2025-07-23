import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from flask import Flask, render_template, request, redirect, url_for, send_file
import cv2
from ultralytics import YOLO, solutions
from werkzeug.utils import secure_filename
import tempfile
import json
import numpy as np
import logging

# Configure logging to reduce noise
logging.getLogger('ultralytics').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.secret_key = 'your-secret-key-here'  # Required for session management

MODEL_PATH = "runs/detect/train/weights/best.pt"
JSON_PATH = "bouding_box.json"

# Global variables to cache the model and parking zones
_model = None
_parking_zones = None

def get_model():
    """Get or load the YOLO model"""
    global _model
    if _model is None:
        if os.path.exists(MODEL_PATH):
            _model = YOLO(MODEL_PATH)
            print(f"Loaded YOLO model: {MODEL_PATH}")
        else:
            print(f"Model file not found: {MODEL_PATH}")
    return _model

def get_parking_zones():
    """Get or load parking zones from JSON"""
    global _parking_zones
    if _parking_zones is None:
        try:
            with open(JSON_PATH, 'r') as f:
                _parking_zones = json.load(f)
            print(f"Loaded {len(_parking_zones)} parking zones from JSON")
        except Exception as e:
            print(f"Error loading parking zones: {e}")
            _parking_zones = []
    return _parking_zones

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    video_url = None
    error = None
    
    if request.method == 'POST':
        try:
            if 'video' not in request.files:
                error = 'No file part'
            else:
                file = request.files['video']
                if file.filename == '':
                    error = 'No selected file'
                elif not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')):
                    error = 'Invalid file format. Please upload a video file.'
                else:
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    
                    # Process video
                    output_path, stats = process_video(filepath)
                    if output_path:
                        video_filename = os.path.basename(output_path)
                        video_url = url_for('download_file', filename=video_filename)
                        result = stats
                    else:
                        error = 'Failed to process video'
        except Exception as e:
            error = f'Error processing video: {str(e)}'
    
    return render_template('index.html', result=result, video_url=video_url, error=error)

@app.route('/download/<filename>')
def download_file(filename):
    try:
        # Security: Only allow downloading files from temp directory
        secure_name = secure_filename(filename)
        if secure_name != filename:
            return "Invalid filename", 400
            
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_name)
        if not os.path.exists(file_path):
            return "File not found", 404
            
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        return f"Error downloading file: {str(e)}", 500

@app.route('/video/<filename>')
def serve_video(filename):
    try:
        # Security: Only allow serving files from temp directory
        secure_name = secure_filename(filename)
        if secure_name != filename:
            return "Invalid filename", 400
            
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_name)
        if not os.path.exists(file_path):
            return "File not found", 404
            
        return send_file(file_path, mimetype='video/mp4')
    except Exception as e:
        return f"Error serving video: {str(e)}", 500

def process_video(input_path):
    """Process video with optimized parking detection"""
    try:
        # Check if input file exists
        if not os.path.exists(input_path):
            print(f"Input file does not exist: {input_path}")
            return None, None
            
        # Get model and parking zones
        model = get_model()
        parking_zones = get_parking_zones()
        
        if model is None:
            print("Could not load model")
            return None, None
            
        if not parking_zones:
            print("No parking zones found")
            return None, None
            
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Could not open video file: {input_path}")
            return None, None
            
        # Get video properties
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Validate video properties
        if w <= 0 or h <= 0 or fps <= 0:
            print(f"Invalid video properties: width={w}, height={h}, fps={fps}")
            cap.release()
            return None, None
            
        print(f"Video properties: {w}x{h}, {fps} FPS, {total_frames} frames")
        
        # Prepare output
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"output_{os.path.basename(input_path)}")
        output_path = output_path.replace('.mp4', '_processed.mp4')
        
        # Use H.264 codec for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        if not video_writer.isOpened():
            print(f"Could not open video writer for: {output_path}")
            cap.release()
            return None, None
        
        # Convert parking zones to numpy arrays for faster processing
        zone_polygons = []
        for zone in parking_zones:
            if 'points' in zone and zone['points']:
                points = np.array(zone['points'], dtype=np.int32)
                zone_polygons.append(points)
        
        total_zones = len(zone_polygons)
        print(f"Processing {total_zones} parking zones")
        
        # Statistics tracking
        frame_count = 0
        total_available = 0
        total_occupied = 0
        
        # Process every frame for better accuracy, but optimize detection
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            try:
                # Run YOLO detection
                results = model(frame, verbose=False)
                
                # Extract detection boxes for cars (assuming class 0 is car)
                car_boxes = []
                if len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    classes = results[0].boxes.cls.cpu().numpy()
                    confidences = results[0].boxes.conf.cpu().numpy()
                    
                    # Filter for cars with good confidence
                    for i, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
                        if conf > 0.5:  # Confidence threshold
                            car_boxes.append(box)
                
                # Check parking zone occupancy
                occupied_zones = 0
                annotated_frame = frame.copy()
                
                for zone_idx, zone_polygon in enumerate(zone_polygons):
                    zone_occupied = False
                    
                    # Check if any car center is in this zone
                    for box in car_boxes:
                        if len(box) >= 4:
                            center_x = int((box[0] + box[2]) / 2)
                            center_y = int((box[1] + box[3]) / 2)
                            
                            # Check if center point is inside the parking zone
                            if cv2.pointPolygonTest(zone_polygon, (center_x, center_y), False) >= 0:
                                zone_occupied = True
                                break
                    
                    # Draw parking zone
                    if zone_occupied:
                        occupied_zones += 1
                        # Draw occupied zone in red
                        cv2.polylines(annotated_frame, [zone_polygon], True, (0, 0, 255), 2)
                        # Add semi-transparent fill
                        overlay = annotated_frame.copy()
                        cv2.fillPoly(overlay, [zone_polygon], (0, 0, 255))
                        annotated_frame = cv2.addWeighted(annotated_frame, 0.8, overlay, 0.2, 0)
                    else:
                        # Draw available zone in green
                        cv2.polylines(annotated_frame, [zone_polygon], True, (0, 255, 0), 2)
                        overlay = annotated_frame.copy()
                        cv2.fillPoly(overlay, [zone_polygon], (0, 255, 0))
                        annotated_frame = cv2.addWeighted(annotated_frame, 0.9, overlay, 0.1, 0)
                
                # Draw car detection boxes
                for box in car_boxes:
                    x1, y1, x2, y2 = map(int, box[:4])
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(annotated_frame, 'Car', (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Calculate current statistics
                available_zones = total_zones - occupied_zones
                total_available += available_zones
                total_occupied += occupied_zones
                
                # Add parking information overlay
                info_text = f"Total: {total_zones} | Available: {available_zones} | Occupied: {occupied_zones}"
                cv2.rectangle(annotated_frame, (10, 10), (900, 50), (0, 0, 0), -1)
                cv2.putText(annotated_frame, info_text, (15, 35), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Add progress indicator
                progress = (frame_count + 1) / total_frames * 100 if total_frames > 0 else 0
                progress_text = f"Processing: {progress:.1f}% ({frame_count + 1}/{total_frames})"
                cv2.putText(annotated_frame, progress_text, (15, h - 20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Write frame
                video_writer.write(annotated_frame)
                
                frame_count += 1
                
                # Print progress every 30 frames
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")
                    
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                # Write original frame on error
                video_writer.write(frame)
                frame_count += 1
                
        cap.release()
        video_writer.release()
        
        # Verify output file
        if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
            print(f"Output file creation failed or file too small")
            return None, None
        
        print(f"Video processing completed: {frame_count} frames processed")
        
        # Calculate final statistics
        if frame_count > 0:
            avg_available = total_available // frame_count
            avg_occupied = total_occupied // frame_count
        else:
            avg_available = 0
            avg_occupied = 0
            
        stats = {
            'total_slots': total_zones,
            'available_slots': avg_available,
            'occupied_slots': avg_occupied,
            'frames_processed': frame_count
        }
        
        print(f"Final statistics: {stats}")
        return output_path, stats
        
    except Exception as e:
        print(f"Error in process_video: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == '__main__':
    # Disable reloader to prevent frequent restarts
    # Only use debug mode for development
    app.run(debug=True, use_reloader=False, host='127.0.0.1', port=5000)
