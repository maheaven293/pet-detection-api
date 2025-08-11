# pet-detection-api
AI-powered pet detection and filtering API with YOLO
## Features

- **Real-time Pet Detection**: YOLO-powered detection of cats, dogs, horses, and more
- **Pet Eye Detection**: Custom-trained YOLO model for precise eye landmark detection
- **Image Filters**: Vintage, sepia, blur, sharpen, brightness adjustments
- **Smart Overlays**: Sunglasses, hats, bow ties positioned using facial landmarks
- **Async Processing**: Non-blocking task queue for heavy operations
- **Cloudflare Tunnel Ready**: Optimized for Raspberry Pi deployment
- **CORS Enabled**: Ready for web frontend integration

## Quick Start

### Prerequisites
```bash
pip install flask flask-cors opencv-python pillow ultralytics mediapipe numpy torch
```

### Installation
```bash
git clone https://github.com/maheaven293/pet-detection-api.git
cd pet-detection-api
python pet detection & filter app
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "pet detection & filter app.py"]
```

## API Endpoints

### Real-time Detection
```http
POST /detect_objects
Content-Type: application/json

{
  "image": "data:image/jpeg;base64,..."
}
```

**Response:**
```json
{
  "boxes": [
    {
      "bbox": [x1, y1, x2, y2],
      "class_name": "cat",
      "confidence": 0.95
    }
  ],
  "pet_type": "cat",
  "count": 1
}
```

### Auto Capture
```http
POST /auto_capture
Content-Type: application/json

{
  "image": "data:image/jpeg;base64,..."
}
```

Returns task ID for async processing.

### Apply Filters
```http
POST /apply_filter
Content-Type: application/json

{
  "image": "data:image/jpeg;base64,...",
  "pet_type": "cat",
  "filter_name": "vintage"
}
```

**Available filters:** `vintage`, `blur`, `sharpen`, `sepia`, `bright`, `dark`

### Apply Overlays
```http
POST /apply_overlay
Content-Type: application/json

{
  "image": "data:image/jpeg;base64,...",
  "overlay_type": "sunglasses"
}
```

**Available overlays:** `sunglasses`, `party_hat`, `bow_tie`

### Facial Landmarks
```http
POST /detect_landmarks
Content-Type: application/json

{
  "image": "data:image/jpeg;base64,..."
}
```

### Task Status
```http
GET /get_task_status/{task_id}
```

## Configuration

### Environment Variables
```bash
PORT=5000
FLASK_DEBUG=false
```

### Model Files Required
- `yolov8n.pt` - Main YOLO model
- `best_pet_eye_yolo.pt` - Custom pet eye detection model

### Directory Structure
```
/static
  /captured_pets     # Auto-captured pet images
  /filtered_images   # Processed results
  /overlay_assets    # PNG overlays (sunglasses.png, etc.)
```

## Performance Optimizations

- **Raspberry Pi Ready**: Reduced image sizes (416px max)
- **Memory Efficient**: Automatic cleanup of old tasks
- **CPU Optimized**: Lower confidence thresholds for faster processing
- **Threading**: Non-blocking async operations

## Frontend Integration

### JavaScript Example
```javascript
const detectPets = async (imageData) => {
  const response = await fetch('http://localhost:5000/detect_objects', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: imageData })
  });
  return await response.json();
};
```

### React Hook
```javascript
const usePetDetection = () => {
  const [pets, setPets] = useState([]);
  
  const detectPets = useCallback(async (imageData) => {
    const result = await fetch('/detect_objects', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageData })
    }).then(r => r.json());
    
    setPets(result.boxes || []);
    return result;
  }, []);
  
  return { pets, detectPets };
};
```

## Deployment Options

### Local Development
```bash
python pet detection & filter app.py
```

### Cloudflare Tunnel
```bash
cloudflared tunnel --url http://localhost:5000
```

### Railway/Heroku
```json
{
  "scripts": {
    "start": "python app.py"
  }
}
```

### Raspberry Pi Setup
```bash
# Install dependencies
sudo apt update
sudo apt install python3-opencv

# Clone and run
git clone https://github.com/maheaven293/pet-detection-api.git
cd pet-detection-api
pip3 install -r requirements.txt
python3 app.py
```

## Custom Model Training

To replace the pet eye detection model:

1. Train YOLO model on pet eye dataset
2. Export to `best_pet_eye_yolo.pt`
3. Adjust confidence thresholds in `detect_pet_eyes_yolo()`

## Error Handling

All endpoints return consistent error format:
```json
{
  "error": "Description of what went wrong",
  "status": "error"
}
```

## Rate Limiting & Security

- Task cleanup every 5 minutes
- 1-hour task expiration
- CORS configured for production
- File size limits (16MB max)

## License

MIT License - Feel free to use in commercial projects.
