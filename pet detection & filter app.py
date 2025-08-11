from flask import Flask, request, jsonify, send_from_directory, render_template, make_response
from flask_cors import CORS
import base64, cv2, io, math, os, threading, time, torch, uuid
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
from ultralytics import YOLO
import mediapipe as mp

app = Flask(__name__)
# Configure CORS for Cloudflare Tunnel
CORS(app, origins=[
    "https://*.trycloudflare.com",  # Cloudflare Tunnel URLs
    "http://localhost:3000",  # Local frontend development
    "http://127.0.0.1:3000",
    "*"  # Allow all origins for Cloudflare Tunnel (since URL changes)
])
app.config['PROXY_FIX'] = True
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False  # Reduce response size
# Global variables
pet_eye_model = None
model = None
task_queue = {}
UPLOAD_FOLDER = 'static'
CAPTURED_PETS_FOLDER = os.path.join(UPLOAD_FOLDER, 'captured_pets')
FILTERED_IMAGES_FOLDER = os.path.join(UPLOAD_FOLDER, 'filtered_images')
OVERLAY_ASSETS_FOLDER = os.path.join(UPLOAD_FOLDER, 'overlay_assets')
# Pet classes that YOLO can detect
PET_CLASSES = {15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe'}

def initialize_pet_eye_model():
    """Initialize pet eye detection YOLO model"""
    global pet_eye_model
    try:
        # Load your trained pet eye model
        pet_eye_model = YOLO('best_pet_eye_yolo.pt')  # Your trained model
        pet_eye_model.to('cpu')
        print("Pet eye YOLO model loaded successfully")
    except Exception as e:
        print(f"Error loading pet eye model: {e}")
        pet_eye_model = None

def detect_pet_eyes_yolo(cropped_face_image):
    """Detect pet eyes using trained YOLO model"""
    if pet_eye_model is None:
        return []
    
    try:
        # Run inference on cropped face
        results = pet_eye_model(np.array(cropped_face_image), conf=0.3, verbose=True)
        
        eyes_detected = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Center point of detected eye
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    eye_info = {
                        'class_id': class_id,
                        'confidence': confidence,
                        'center': (center_x, center_y),
                        'bbox': [int(x1), int(y1), int(x2), int(y2)]
                    }
                    eyes_detected.append(eye_info)
        
        return eyes_detected
        
    except Exception as e:
        print(f"Error in YOLO eye detection: {e}")
        return []

def initialize_model():
    """Initialize main pet detection YOLO model"""
    global model
    model = YOLO('yolov8n.pt')
    model.to('cpu')
    print("Main YOLO model loaded")

def initialize_pet_eye_model():
    """Initialize pet eye detection YOLO model"""
    global pet_eye_model
    pet_eye_model = YOLO('best_pet_eye_yolo.pt')
    pet_eye_model.to('cpu')
    print("Pet eye YOLO model loaded")
    
def initialize_models():
    initialize_model()
    initialize_pet_eye_model()

def ensure_directories():
    [os.makedirs(folder, exist_ok=True) for folder in [UPLOAD_FOLDER, CAPTURED_PETS_FOLDER, FILTERED_IMAGES_FOLDER, OVERLAY_ASSETS_FOLDER]]
    
def base64_to_image(base64_string):
    """Convert base64 string to PIL Image"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        # Decode base64
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        print(f"Error converting base64 to image: {e}")
        return None

def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    try:
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None

def detect_pets_in_image(image):
    """Detect pets in image using YOLO - optimized for Raspberry Pi"""
    if model is None:
        return [], None
    try:
        original_width, original_height = image.size
        max_size = 416  # Smaller image size for faster Pi performance
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple([int(x * ratio) for x in image.size])
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        img_array = np.array(image)
        results = model(img_array, imgsz=416, conf=0.4, verbose=False)  # Lower confidence, smaller image size
        pets_detected = []
        largest_pet = None
        max_area = 0
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    # Check if it's a pet class and confidence is high enough
                    if class_id in PET_CLASSES and confidence > 0.4:  # Lower threshold for Pi
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        pet_info = {'class_id': class_id, 'class_name': PET_CLASSES[class_id], 'confidence': confidence, 'bbox': [int(x1), int(y1), int(x2), int(y2)]}
                        pets_detected.append(pet_info)
                        # Track largest pet
                        area = (x2 - x1) * (y2 - y1)
                        if area > max_area:
                            max_area = area
                            largest_pet = pet_info
        if max(original_width, original_height) > max_size:
            scale_factor = max(original_width, original_height) / max_size
            for pet_info in pets_detected:
                bbox = pet_info['bbox']
                pet_info['bbox'] = [int(coord * scale_factor) for coord in bbox]
        return pets_detected, largest_pet
    except Exception as e:
        print(f"Error in pet detection: {e}")
        return [], None

def apply_filter_to_image(image, filter_name, pet_type):
    """Apply filter to image based on pet type and filter name"""
    try:
        if filter_name == "vintage":
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(0.7)  # Reduce saturation
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)  # Increase contrast
        elif filter_name == "blur":
            image = image.filter(ImageFilter.GaussianBlur(radius=2))
        elif filter_name == "sharpen":
            image = image.filter(ImageFilter.SHARPEN)
        elif filter_name == "sepia":
            pixels = image.load()
            for i in range(image.width):
                for j in range(image.height):
                    r, g, b = pixels[i, j][:3]  # Handle RGBA by taking first 3 values
                    tr = int(0.393 * r + 0.769 * g + 0.189 * b)
                    tg = int(0.349 * r + 0.686 * g + 0.168 * b)
                    tb = int(0.272 * r + 0.534 * g + 0.131 * b)
                    pixels[i, j] = (min(255, tr), min(255, tg), min(255, tb))
        elif filter_name == "bright":
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.3)
        elif filter_name == "dark":
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(0.7)
        return image
    except Exception as e:
        print(f"Error applying filter: {e}")
        return image

def perform_auto_capture_task(task_id, image_data):
    """Asynchronous task for auto capture"""
    try:
        # Convert image to base64 so it can be processed
        original_image = base64_to_image(image_data)
        if original_image is None:
            task_queue[task_id]['status'] = 'error'
            task_queue[task_id]['error'] = 'Failed to decode image'
            return
        # Store original dimensions to prevent bbox offset
        original_width = original_image.width
        original_height = original_image.height        
        pets_detected, largest_pet = detect_pets_in_image(original_image)
        if largest_pet is None:
            task_queue[task_id]['status'] = 'error'
            task_queue[task_id]['error'] = 'No pets detected'
            return
        # Crop image to pet bounding box with some padding
        bbox = largest_pet['bbox']
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        padding_x = int(width * 0.1)
        padding_y = int(height * 0.1)
        x1 = max(0, x1 - padding_x)
        y1 = max(0, y1 - padding_y)
        x2 = min(original_image.width, x2 + padding_x)
        y2 = min(original_image.height, y2 + padding_y)
        cropped_image = original_image.crop((x1, y1, x2, y2))
        #filename = f"captured_pet_{task_id}.jpg"
        #filepath = os.path.join(CAPTURED_PETS_FOLDER, filename)
        #cropped_image.save(filepath)
        # Convert cropped image back to base64 for frontend display
        captured_image_b64 = image_to_base64(cropped_image)
        task_queue[task_id]['status'] = 'completed'
        #, 'filepath': filepath
        task_queue[task_id]['result'] = { 'captured_image': captured_image_b64, 'pet_type': largest_pet['class_name'], 'confidence': largest_pet['confidence'], 'boxes': [largest_pet['bbox']]}
    except Exception as e:
        print(f"Error in auto capture task: {e}")
        task_queue[task_id]['status'] = 'error'
        task_queue[task_id]['error'] = str(e)

def perform_apply_filter_task(task_id, image_data, pet_type, filter_name):
    """Asynchronous task for applying filter"""
    try:
        image = base64_to_image(image_data)
        if image is None:
            task_queue[task_id]['status'] = 'error'
            task_queue[task_id]['error'] = 'Failed to decode image'
            return
        filtered_image = apply_filter_to_image(image, filter_name, pet_type)
        # Save filtered image to existing folder in server host, make sure the folder exists 
        filename = f"filtered_{task_id}_{filter_name}.jpg"
        filepath = os.path.join(FILTERED_IMAGES_FOLDER, filename)
        filtered_image.save(filepath)
        task_queue[task_id]['status'] = 'completed'
        task_queue[task_id]['result'] = {'filtered_image_url': f"/static/filtered_images/{filename}", 'filter_applied': filter_name, 'pet_type': pet_type}
    except Exception as e:
        print(f"Error in apply filter task: {e}")
        task_queue[task_id]['status'] = 'error'
        task_queue[task_id]['error'] = str(e)

def detect_pet_face_landmarks_yolo(image):
    """New YOLO-based pet face landmark detection"""
    try:
        img_array = np.array(image)
        print(f"YOLO eye detection - Image size: {image.size}")
        
        # Detect eyes using trained YOLO model
        eyes_detected = detect_pet_eyes_yolo(image)
        print(f"YOLO detected {len(eyes_detected)} eyes")
        
        if len(eyes_detected) >= 2:
            # Find valid eye pairs (horizontally aligned)
            valid_pairs = []
            for i, eye1 in enumerate(eyes_detected):
                for j, eye2 in enumerate(eyes_detected[i+1:], i+1):
                    dy = abs(eye1['center'][1] - eye2['center'][1])
                    if dy < 30:  # Eyes should be roughly same height
                        valid_pairs.append((eye1, eye2))
            
            if valid_pairs:
                # Take pair with highest combined confidence
                best_pair = max(valid_pairs, key=lambda p: p[0]['confidence'] + p[1]['confidence'])
                eye1, eye2 = best_pair
            else:
                # Fallback to top 2 by confidence if no valid pairs
                eyes_detected.sort(key=lambda x: x['confidence'], reverse=True)
                eye1, eye2 = eyes_detected[0], eyes_detected[1]
            
            # Determine left/right based on x-coordinate
            if eye1['center'][0] < eye2['center'][0]:
                left_eye = eye1['center']
                right_eye = eye2['center']
            else:
                left_eye = eye2['center']
                right_eye = eye1['center']
            
            print(f"YOLO eyes: left={left_eye}, right={right_eye}")
            
            # Calculate face bbox from eye positions
            eye_dist = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
            face_width = eye_dist * 2.5
            face_height = eye_dist * 2.0
            
            face_center_x = (left_eye[0] + right_eye[0]) // 2
            face_center_y = (left_eye[1] + right_eye[1]) // 2
            
            face_x = int(face_center_x - face_width / 2)
            face_y = int(face_center_y - face_height / 2)
            
            landmarks = [{
                'face_bbox': [face_x, face_y, face_x + int(face_width), face_y + int(face_height)],
                'estimated_left_eye': list(left_eye),
                'estimated_right_eye': list(right_eye),
                'estimated_nose': [face_center_x, face_center_y + int(eye_dist * 0.5)],
                'estimated_mouth': [face_center_x, face_center_y + int(eye_dist * 1.0)],
                'keypoints': {},
                'confidence': min(eye1['confidence'], eye2['confidence']),
                'method': 'yolo_pet_eyes',
                'detected_eyes': eyes_detected
            }]
            
            return landmarks
        
        else:
            print("YOLO: Insufficient eyes detected, using fallback")
            return detect_pet_features_alternative(img_array)
            
    except Exception as e:
        print(f"Error in YOLO eye detection: {e}")
        return detect_pet_features_alternative(np.array(image))

def apply_overlay_to_pet(image, landmarks, overlay_type):
    """Apply overlay (sunglasses, hat, etc.) to pet face"""
    try:
        if not landmarks:
            return image
            
        # Get the first (largest) face landmarks
        face_data = landmarks[0]
        
        # Create overlay based on type
        overlay_img = create_overlay_image(overlay_type, face_data)
        if overlay_img is None:
            return image
            
        # Position overlay on face
        positioned_overlay = position_overlay_on_face(overlay_img, face_data, image.size, overlay_type)
        
        # Composite images
        if positioned_overlay:
            # Convert to RGBA for transparency
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
            if positioned_overlay.mode != 'RGBA':
                positioned_overlay = positioned_overlay.convert('RGBA')
                
            # Paste overlay onto image
            image.paste(positioned_overlay, (0, 0), positioned_overlay)
            
        return image.convert('RGB')  # Convert back to RGB
        
    except Exception as e:
        print(f"Error applying overlay: {e}")
        return image

def create_overlay_image(overlay_type, face_data):
    """Load and scale premade overlay images"""
    try:
        bbox = face_data['face_bbox']
        face_width = bbox[2] - bbox[0]
        face_height = bbox[3] - bbox[1]
        
        # Map overlay types to file paths
        overlay_files = {
            'sunglasses': os.path.join(OVERLAY_ASSETS_FOLDER, 'sunglasses.png'),
            'party_hat': os.path.join(OVERLAY_ASSETS_FOLDER, 'party_hat.png'),
            'bow_tie': os.path.join(OVERLAY_ASSETS_FOLDER, 'bow_tie.png')
        }
        
        if overlay_type not in overlay_files:
            print(f"Unknown overlay type: {overlay_type}")
            return None
            
        overlay_path = overlay_files[overlay_type]
        
        # Check if file exists
        if not os.path.exists(overlay_path):
            print(f"Overlay file not found: {overlay_path}")
            return None
            
        # Load premade overlay image
        overlay_img = Image.open(overlay_path).convert('RGBA')
        
        # Scale overlay to appropriate size based on face dimensions
        if overlay_type == 'sunglasses':
            # Scale sunglasses to 80% of face width
            target_width = int(face_width * 0.9)
            aspect_ratio = overlay_img.height / overlay_img.width
            target_height = int(target_width * aspect_ratio)
            
        elif overlay_type == 'party_hat':
            # Scale hat to 60% of face width, maintain aspect ratio
            target_width = int(face_width * 0.5)
            aspect_ratio = overlay_img.height / overlay_img.width
            target_height = int(target_width * aspect_ratio)
            
        elif overlay_type == 'bow_tie':
            # Scale bow tie to 40% of face width
            target_width = int(face_width * 0.5)
            aspect_ratio = overlay_img.height / overlay_img.width
            target_height = int(target_width * aspect_ratio)
        
        else:
            # Default scaling
            target_width = int(face_width * 0.6)
            aspect_ratio = overlay_img.height / overlay_img.width
            target_height = int(target_width * aspect_ratio)
            
        # Resize overlay image
        scaled_overlay = overlay_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        return scaled_overlay
        
    except Exception as e:
        print(f"Error loading overlay image: {e}")
        return None

def position_overlay_on_face(overlay_img, face_data, image_size, overlay_type):
    """Position overlay correctly on the pet's face"""
    try:
        bbox = face_data['face_bbox']
        face_width = bbox[2] - bbox[0]
        face_height = bbox[3] - bbox[1]
        
        # Create full-size overlay image
        full_overlay = Image.new('RGBA', image_size, (0, 0, 0, 0))
        
        # Get estimated eye position from landmarks if available
        if 'estimated_left_eye' in face_data and 'estimated_right_eye' in face_data:
            left_eye = face_data['estimated_left_eye']
            right_eye = face_data['estimated_right_eye']
            eye_center_y = (left_eye[1] + right_eye[1]) // 2
            eye_center_x = (left_eye[0] + right_eye[0]) // 2
        else:
            # Fallback: assume eyes are in upper third of face
            eye_center_x = bbox[0] + face_width // 2
            eye_center_y = bbox[1] + face_height // 3
        
        if overlay_type == 'sunglasses':
            # Center sunglasses on the detected/estimated eyes
            overlay_x = eye_center_x - overlay_img.width // 2
            overlay_y = eye_center_y - overlay_img.height // 2
            
        elif overlay_type == 'party_hat':
            # Position hat above the eyes, scaled to face size
            overlay_x = eye_center_x - overlay_img.width // 2
            # Place hat so bottom edge is just above the eyes
            overlay_y = eye_center_y - int(face_height * 0.6) - overlay_img.height
            overlay_y = max(0, overlay_y)  # Keep within image bounds
            
        elif overlay_type == 'bow_tie':
            # Position bow tie at estimated chin/neck area
            # Use nose position if available, otherwise estimate
            if 'estimated_nose' in face_data:
                nose_y = face_data['estimated_nose'][1]
                overlay_y = nose_y + int(face_height * 0.3)
            else:
                overlay_y = bbox[1] + int(face_height * 0.7)
            overlay_x = eye_center_x - overlay_img.width // 2
            
        else:
            # Default positioning
            overlay_x = bbox[0]
            overlay_y = bbox[1]
        
        # Ensure overlay stays within image bounds
        overlay_x = max(0, min(overlay_x, image_size[0] - overlay_img.width))
        overlay_y = max(0, min(overlay_y, image_size[1] - overlay_img.height))
        
        # Paste overlay at calculated position
        full_overlay.paste(overlay_img, (int(overlay_x), int(overlay_y)), overlay_img)
        
        return full_overlay
        
    except Exception as e:
        print(f"Error positioning overlay: {e}")
        return None

def perform_apply_overlay_task(task_id, image_data, overlay_type):
    """Asynchronous task for applying overlay"""
    try:
        image = base64_to_image(image_data)
        if image is None:
            task_queue[task_id]['status'] = 'error'
            task_queue[task_id]['error'] = 'Failed to decode image'
            return
        
        # Add debug logging
        print(f"Image size for overlay: {image.size}")
        
        # Detect face landmarks first
        landmarks = detect_pet_face_landmarks_yolo(image)
        if not landmarks:
            task_queue[task_id]['status'] = 'error'
            task_queue[task_id]['error'] = 'No face detected for overlay'
            return
        # Log detected landmarks
        print(f"Detected landmarks using {landmarks[0]['method']}: {landmarks[0]['estimated_left_eye']}, {landmarks[0]['estimated_right_eye']}")
        # Apply overlay
        overlay_image = apply_overlay_to_pet(image, landmarks, overlay_type)
        # Save overlaid image
        filename = f"overlay_{task_id}_{overlay_type}.jpg"
        filepath = os.path.join(FILTERED_IMAGES_FOLDER, filename)
        overlay_image.save(filepath)
        task_queue[task_id]['status'] = 'completed'
        task_queue[task_id]['result'] = {
            'overlay_image_url': f"/static/filtered_images/{filename}",
            'overlay_applied': overlay_type,
            'landmarks_detected': len(landmarks)
        }
    except Exception as e:
        print(f"Error in apply overlay task: {e}")
        task_queue[task_id]['status'] = 'error'
        task_queue[task_id]['error'] = str(e)
            
def detect_pet_features_alternative(img_array):
    """Alternative pet feature detection using conservative estimates"""
    try:
        h, w = img_array.shape[:2] if len(img_array.shape) >= 2 else (100, 100)
        
        # Conservative approach: assume face is in center-upper region
        # More suitable for already-cropped pet images
        face_x = int(w * 0.2)
        face_y = int(h * 0.1)
        face_width = int(w * 0.6)
        face_height = int(h * 0.5)
        
        # Conservative eye positions for cats
        left_eye = [face_x + int(face_width * 0.3), face_y + int(face_height * 0.35)]
        right_eye = [face_x + int(face_width * 0.7), face_y + int(face_height * 0.35)]
        
        landmarks = [{
            'face_bbox': [face_x, face_y, face_x + face_width, face_y + face_height],
            'estimated_left_eye': left_eye,
            'estimated_right_eye': right_eye,
            'estimated_nose': [face_x + int(face_width * 0.5), face_y + int(face_height * 0.55)],
            'estimated_mouth': [face_x + int(face_width * 0.5), face_y + int(face_height * 0.75)],
            'keypoints': {},
            'confidence': 0.5,
            'method': 'geometric_fallback'
        }]
        
        return landmarks
    except Exception as e:
        print(f"Error in alternative detection: {e}")
        return []

def perform_landmark_detection_task(task_id, image_data):
    """Updated landmark detection using YOLO eyes"""
    try:
        image = base64_to_image(image_data)
        if image is None:
            task_queue[task_id]['status'] = 'error'
            task_queue[task_id]['error'] = 'Failed to decode image'
            return
        landmarks = detect_pet_face_landmarks_yolo(image)
        if not landmarks:
            task_queue[task_id]['status'] = 'error'
            task_queue[task_id]['error'] = 'No facial features detected'
            return
        task_queue[task_id]['status'] = 'completed'
        task_queue[task_id]['result'] = { 'landmarks': landmarks, 'face_detected': len(landmarks) > 0, 'landmark_count': len(landmarks), 'method': landmarks[0]['method']}
    except Exception as e:
        print(f"Error in landmark detection task: {e}")
        task_queue[task_id]['status'] = 'error'
        task_queue[task_id]['error'] = str(e)

def cleanup_old_tasks():
    current_time = time.time()
    expired_tasks = [task_id for task_id, task in task_queue.items() 
                     if current_time - task.get('created_at', 0) > 3600]
    for task_id in expired_tasks:
        del task_queue[task_id]

def cleanup_old_tasks_periodically():
    while True:
        time.sleep(300)  # Clean every 5 minutes
        cleanup_old_tasks()

# Start cleanup thread
cleanup_thread = threading.Thread(target=cleanup_old_tasks_periodically)
cleanup_thread.daemon = True
cleanup_thread.start()
# Route statements
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    """Real-time object detection endpoint"""
    try:
        data = request.get_json()
        image_data = data.get('image')
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        image = base64_to_image(image_data)
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400
        pets_detected, largest_pet = detect_pets_in_image(image)
        boxes = []
        pet_type = None
        for pet in pets_detected:
            boxes.append({ 'bbox': pet['bbox'], 'class_name': pet['class_name'], 'confidence': pet['confidence']})
        if largest_pet:
            pet_type = largest_pet['class_name']
        return jsonify({ 'boxes': boxes, 'pet_type': pet_type, 'count': len(pets_detected)})
    except Exception as e:
        print(f"Error in detect_objects: {e}")
        return jsonify({'error': 'Detection failed'}), 500

@app.route('/auto_capture', methods=['POST'])
def auto_capture():
    """Start asynchronous auto capture task"""
    try:
        data = request.get_json()
        image_data = data.get('image')
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        task_id = str(uuid.uuid4())
        task_queue[task_id] = {'status': 'processing', 'created_at': time.time()}
        thread = threading.Thread(target=perform_auto_capture_task, args=(task_id, image_data))
        thread.daemon = True
        thread.start()
        return jsonify({'status': 'processing', 'task_id': task_id})
    except Exception as e:
        print(f"Error in auto_capture: {e}")
        return jsonify({'error': 'Capture failed'}), 500

@app.route('/apply_filter', methods=['POST'])
def apply_filter():
    """Start asynchronous filter application task"""
    try:
        data = request.get_json()
        image_data = data.get('image')
        pet_type = data.get('pet_type')
        filter_name = data.get('filter_name')
        if not all([image_data, pet_type, filter_name]):
            return jsonify({'error': 'Missing required parameters'}), 400
        task_id = str(uuid.uuid4())
        task_queue[task_id] = {'status': 'processing', 'created_at': time.time()}
        thread = threading.Thread(target=perform_apply_filter_task, args=(task_id, image_data, pet_type, filter_name))
        thread.daemon = True
        thread.start()
        return jsonify({'status': 'processing', 'task_id': task_id})
    except Exception as e:
        print(f"Error in apply_filter: {e}")
        return jsonify({'error': 'Filter application failed'}), 500

@app.route('/detect_landmarks', methods=['POST'])
def detect_landmarks():
    """Pet face landmark detection endpoint"""
    try:
        data = request.get_json()
        image_data = data.get('image')
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        task_id = str(uuid.uuid4())
        task_queue[task_id] = {'status': 'processing', 'created_at': time.time()}
        thread = threading.Thread(target=perform_landmark_detection_task,args=(task_id, image_data))
        thread.daemon = True
        thread.start()
        return jsonify({'status': 'processing', 'task_id': task_id})
    except Exception as e:
        print(f"Error in detect_landmarks: {e}")
        return jsonify({'error': 'Landmark detection failed'}), 500

@app.route('/apply_overlay', methods=['POST'])
def apply_overlay():
    """Start asynchronous overlay application task"""
    try:
        data = request.get_json()
        image_data = data.get('image')
        overlay_type = data.get('overlay_type')
        
        if not all([image_data, overlay_type]):
            return jsonify({'error': 'Missing required parameters'}), 400
            
        task_id = str(uuid.uuid4())
        task_queue[task_id] = {'status': 'processing', 'created_at': time.time()}
        
        thread = threading.Thread(target=perform_apply_overlay_task, 
                                args=(task_id, image_data, overlay_type))
        thread.daemon = True
        thread.start()
        
        return jsonify({'status': 'processing', 'task_id': task_id})
        
    except Exception as e:
        print(f"Error in apply_overlay: {e}")
        return jsonify({'error': 'Overlay application failed'}), 500

@app.route('/apply_both', methods=['POST'])
def apply_both():
    """Apply both filter and overlay to image"""
    try:
        data = request.get_json()
        image_data = data.get('image')
        filter_name = data.get('filter_name')
        overlay_type = data.get('overlay_type')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
            
        # Apply filter first if specified
        image = base64_to_image(image_data)
        if image is None:
            return jsonify({'error': 'Failed to decode image'}), 400        
        pet_type = data.get('pet_type')
        if filter_name:
            image = apply_filter_to_image(image, filter_name, pet_type)
            
        # Then apply overlay if specified
        if overlay_type:
            landmarks = detect_pet_face_landmarks_yolo(image)
            image = apply_overlay_to_pet(image, landmarks, overlay_type)
            
        # Save and return result
        filename = f"combined_{uuid.uuid4()}.jpg"
        filepath = os.path.join(FILTERED_IMAGES_FOLDER, filename)
        image.save(filepath)
        
        return jsonify({
            'status': 'success',
            'image_url': f"/static/filtered_images/{filename}",
            'filter_applied': filter_name,
            'overlay_applied': overlay_type
        })
        
    except Exception as e:
        print(f"Error in apply_both: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_task_status/<task_id>')
def get_task_status(task_id):
    """Get status of asynchronous task"""
    cleanup_old_tasks()
    if task_id not in task_queue:
        return jsonify({'error': 'Task not found'}), 404
    task = task_queue[task_id]
    response = {'status': task['status'], 'task_id': task_id}
    if task['status'] == 'completed':
        response['result'] = task.get('result', {})
    elif task['status'] == 'error':
        response['error'] = task.get('error', 'Unknown error')
    return jsonify(response)

@app.route('/reset_capture', methods=['POST'])
def reset_capture():
    """Reset capture state and clean up files"""
    try:
        # Clean up old task queue entries (optional)
        current_time = time.time()
        old_tasks = [task_id for task_id, task in task_queue.items() if current_time - task.get('created_at', 0) > 3600]
        for task_id in old_tasks:
            del task_queue[task_id]
        return jsonify({'status': 'success', 'message': 'Capture reset successfully'})
    except Exception as e:
        print(f"Error in reset_capture: {e}")
        return jsonify({'error': 'Reset failed'}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    response = make_response(send_from_directory('static', filename))
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@app.route('/static/captured_pets/<path:filename>')
def serve_captured_pets(filename):
    """Serve captured pet images"""
    return send_from_directory(CAPTURED_PETS_FOLDER, filename)

@app.route('/static/filtered_images/<path:filename>')
def serve_filtered_images(filename):
    """Serve filtered images"""
    return send_from_directory(FILTERED_IMAGES_FOLDER, filename)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None, 'active_tasks': len(task_queue)})

if __name__ == '__main__':
    print("Initializing Pet Filter Backend for Raspberry Pi + Cloudflare Tunnel...")
    ensure_directories()
    initialize_models()
    print("Starting Flask application...")
    # Configuration optimized for Raspberry Pi + Cloudflare Tunnel
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 for your tunnel setup
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    # Raspberry Pi optimizations
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 300  # 5 minute cache for static files
    print(f"Flask app starting on port {port}")
    print("Waiting for Cloudflare Tunnel to establish connection...")
    app.run(debug=debug_mode, host='127.0.0.1', port=port, threaded=True, use_reloader=False)
