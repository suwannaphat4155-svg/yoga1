# Yoga Pose Detection with YOLO

A Flask-based web application for real-time yoga pose detection using YOLOv8. Upload images to detect and annotate yoga poses with confidence scores.

## Features

✅ **Real-time Pose Detection** - Uses YOLOv8 for accurate pose detection  
✅ **Web Interface** - Simple drag-and-drop image upload  
✅ **Image Annotation** - Visual feedback with bounding boxes  
✅ **Confidence Scores** - Detailed detection results with accuracy metrics  
✅ **Serverless Deployment** - Ready for Vercel deployment  

## Quick Start

### Prerequisites
- Python 3.9 or higher
- pip or conda

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd yoga
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run locally**
```bash
python app.py
```

Open your browser and navigate to `http://localhost:5000`

## Deployment

### Deploy to Vercel

**Option 1: Using Vercel CLI**
```bash
npm install -g vercel
vercel login
vercel --prod
```

**Option 2: GitHub Integration (Recommended)**
1. Push code to GitHub
2. Go to [Vercel](https://vercel.com)
3. Click "New Project"
4. Select your GitHub repository
5. Click "Deploy"

See [DEPLOYMENT.md](./DEPLOYMENT.md) for detailed instructions.

## Project Structure

```
├── api/
│   └── index.py              # Flask app (Vercel entry point)
├── templates/
│   └── index.html            # Frontend UI
├── app.py                    # Original Flask app (for local dev)
├── best.pt                   # YOLOv8 model file
├── requirements.txt          # Python dependencies
├── vercel.json              # Vercel configuration
├── .vercelignore            # Vercel build ignore rules
├── .gitignore               # Git ignore rules
├── DEPLOYMENT.md            # Deployment guide
└── README.md                # This file
```

## API Endpoints

### `GET /`
Returns the web interface (index.html)

### `POST /predict_ui`
Detects poses and returns annotated image
- **Input**: Image file (multipart/form-data)
- **Output**: JPEG image with annotations

Example:
```bash
curl -X POST -F "file=@image.jpg" http://localhost:5000/predict_ui --output result.jpg
```

### `POST /predict_json`
Detects poses and returns JSON results
- **Input**: Image file (multipart/form-data)
- **Output**: JSON with detection results

Example:
```bash
curl -X POST -F "file=@image.jpg" http://localhost:5000/predict_json | jq
```

Response:
```json
{
  "accuracy": 85.5,
  "detections": [
    {
      "label": "person",
      "confidence": 92.3,
      "box": [100, 150, 400, 600]
    }
  ],
  "pose_count": 1,
  "status": "success"
}
```

## Configuration

### Model File
- **File**: `best.pt` (YOLOv8 model)
- **Size**: Check your model size (ensure < 250MB for Vercel)
- **Location**: Root directory

### Environment Variables (Optional)
```bash
PORT=5000                    # Port for local development
PYTHONUNBUFFERED=1          # Python output buffering (set in Vercel)
```

## Development

### Running Tests Locally

```bash
# Install test dependencies
pip install pytest

# Run tests
pytest
```

### Making Changes

1. Create a new branch
```bash
git checkout -b feature/your-feature
```

2. Make your changes
3. Test locally with `vercel dev`
4. Push and create a Pull Request

## Troubleshooting

### Model Loading Issues
- Ensure `best.pt` exists in the root directory
- Check file permissions
- Verify the file is not corrupted

### Memory Issues
- Consider using a smaller YOLO model (nano/small)
- Optimize the model using quantization

### Slow Response Time
- Cold starts are normal for serverless functions
- Optimize model inference with batch processing

### File Upload Errors
- Check file size limits (typically 4.5MB on Vercel)
- Verify image format is supported (JPG, PNG, etc.)

## Performance Tips

1. **Image Optimization**: Resize images before upload for faster processing
2. **Model Caching**: Model is cached per function instance
3. **Batch Processing**: Consider processing multiple images for better throughput

## Future Improvements

- [ ] Support for video input
- [ ] Real-time pose tracking with multiple poses
- [ ] Custom model support
- [ ] Improved UI/UX
- [ ] Caching for repeated predictions
- [ ] Model quantization for faster inference

## License

MIT License - feel free to use this project for your needs

## Support

For issues and questions:
- Check [DEPLOYMENT.md](./DEPLOYMENT.md) for deployment-specific help
- Review the [Vercel Python documentation](https://vercel.com/docs/concepts/functions/serverless-functions/python)
- Open an issue on GitHub

## References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Vercel Documentation](https://vercel.com/docs)
