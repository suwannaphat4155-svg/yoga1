# Yoga Pose Detection - Vercel Deployment Guide

## Project Structure
```
yoga/
├── api/
│   └── index.py          # Main Flask app (Vercel entry point)
├── templates/
│   └── index.html        # Frontend UI
├── best.pt               # YOLO model
├── requirements.txt      # Python dependencies
├── vercel.json          # Vercel configuration
├── .vercelignore        # Files to ignore during deployment
└── .gitignore           # Git ignore rules
```

## Prerequisites
- [Vercel CLI](https://vercel.com/docs/cli) installed (`npm i -g vercel`)
- Git repository initialized
- Python 3.9+ (for local testing)

## Local Testing

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run locally
```bash
python app.py
```
Visit `http://localhost:5000` in your browser.

### 3. Test with Vercel CLI locally
```bash
vercel dev
```

## Deployment to Vercel

### Option 1: Using Vercel CLI
```bash
vercel login
vercel --prod
```

### Option 2: Using GitHub Integration (Recommended)
1. Push your code to GitHub
2. Go to https://vercel.com
3. Click "New Project"
4. Import your GitHub repository
5. Vercel will automatically detect the `vercel.json` configuration
6. Click "Deploy"

## Important Notes

### Model File (best.pt)
- The model file is included in the repository
- For production, ensure `best.pt` is committed to git
- File size limits: Vercel projects can handle up to 250MB
- If your model is too large, consider:
  - Using model compression/quantization
  - Hosting the model separately and downloading it at runtime

### Environment Variables
Add these if needed in Vercel Project Settings:
- `PYTHONUNBUFFERED=1` (already set in vercel.json)

### Cold Start Performance
- First request may take longer due to model loading
- Consider adding timeout handling on frontend

### Troubleshooting

**"Module not found" errors:**
- Ensure `requirements.txt` includes all dependencies
- Check that the file path to `best.pt` is correct

**Model loading fails:**
- Verify `best.pt` is in the root directory
- Check file permissions

**File upload issues:**
- Max file size depends on Vercel plan
- Default is typically 4.5MB for request body

## API Endpoints

### GET `/`
Returns the HTML interface

### POST `/predict_ui`
- Upload an image file
- Returns annotated image with detections
- Content-Type: multipart/form-data

### POST `/predict_json`
- Upload an image file
- Returns JSON with detection results and confidence scores
- Response format:
```json
{
  "accuracy": 85.5,
  "detections": [
    {
      "label": "person",
      "confidence": 92.3,
      "box": [x1, y1, x2, y2]
    }
  ],
  "pose_count": 1,
  "status": "success"
}
```

## Monitoring & Logs
View deployment logs in Vercel dashboard:
- https://vercel.com/dashboard
- Select your project
- Go to "Deployments" tab
- Click on a deployment to view logs

## Performance Optimization Tips

1. **Model Optimization:**
   - Use model quantization to reduce size
   - Consider using smaller YOLO versions (nano, small)

2. **Caching:**
   - Model is loaded once per function instance
   - Serverless instances are reused for better performance

3. **Image Processing:**
   - Current implementation uses Pillow for image handling
   - Works well for typical image sizes

## Common Commands

```bash
# Deploy to production
vercel --prod

# Deploy preview
vercel

# View logs
vercel logs

# Remove a deployment
vercel remove [URL]
```

## Resources
- [Vercel Python Runtime](https://vercel.com/docs/concepts/functions/serverless-functions/python)
- [Vercel Documentation](https://vercel.com/docs)
- [Flask on Vercel](https://vercel.com/guides/deploying-flask-with-vercel)
