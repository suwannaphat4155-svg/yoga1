from flask import Flask, request, render_template, send_file, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
import os
import gdown

app = Flask(__name__, template_folder="templates")

# Load model from Google Drive
def load_model():
    model_path = "best.pt"
    google_drive_id = "1lAz1yd2zufeq0GKmFNAmrPQkCB2Qdnv3"
    
    # Download if not exists
    if not os.path.exists(model_path):
        print("📥 Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={google_drive_id}"
        gdown.download(url, model_path, quiet=False)
        print("✓ Model downloaded successfully")
    
    model = YOLO(model_path)
    print("✓ Model loaded successfully")
    return model

model = load_model()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict_ui", methods=["POST"])
def predict_ui():
    try:
        if "file" not in request.files:
            return "No file", 400

        file = request.files["file"]
        
        # Handle both file and blob uploads
        try:
            img = Image.open(file.stream)
        except:
            file.seek(0)
            img = Image.open(io.BytesIO(file.read()))
        
        # Convert to RGB and resize for processing
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Resize for consistent processing (maintain aspect ratio)
        img.thumbnail((1280, 720), Image.Resampling.LANCZOS)
        
        print(f"Processing image size: {img.size}")
        results = model(img, conf=0.25, verbose=False)  # Lower threshold for better detection
        print(f"Detections found: {len(results[0].boxes) if results and len(results) > 0 else 0}")
        
        if not results or len(results) == 0:
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            buf.seek(0)
            return send_file(buf, mimetype="image/jpeg")
        
        annotated = results[0].plot()
        im = Image.fromarray(annotated)
        buf = io.BytesIO()
        im.save(buf, format="JPEG")
        buf.seek(0)

        return send_file(buf, mimetype="image/jpeg")
    except Exception as e:
        print(f"Error in predict_ui: {str(e)}")
        return f"Error: {str(e)}", 500


@app.route("/predict_json", methods=["POST"])
def predict_json():
    """Return pose detection with confidence scores for accuracy percentage"""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file", "accuracy": 0, "detections": []}), 400

        file = request.files["file"]
        
        # Handle both file and blob uploads
        try:
            img = Image.open(file.stream)
        except:
            file.seek(0)
            img = Image.open(io.BytesIO(file.read()))
        
        # Convert to RGB and resize for faster processing
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Resize for consistent processing
        img.thumbnail((1280, 720), Image.Resampling.LANCZOS)
        
        # Run detection with lower confidence for better results
        results = model(img, conf=0.2, verbose=False)
        print(f"[Detection] Got results: {type(results)}, len: {len(results) if results else 0}")
        
        if not results or len(results) == 0:
            print("[Detection] No detections found")
            return jsonify({
                "accuracy": 0,
                "detections": [],
                "pose_count": 0,
                "status": "no_detection"
            })
        
        result = results[0]
        print(f"[Detection] Processing result - boxes: {len(result.boxes) if result.boxes else 0}")
        
        # Extract confidence scores and labels
        detections = []
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                conf = float(box.conf[0]) * 100  # Convert to percentage
                cls = int(box.cls[0])
                label = None
                try:
                    label = model.names[cls]
                except Exception:
                    label = str(cls)
                detections.append({
                    "class": cls,
                    "label": label,
                    "confidence": round(conf, 2),
                    "bbox": box.xyxy[0].tolist()
                })

        accuracy = round(np.mean([d["confidence"] for d in detections]), 2) if detections else 0
        pose_label = "ไม่พบท่า"
        if detections:
            sorted_detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)
            pose_label = sorted_detections[0]["label"]

        return jsonify({
            "accuracy": accuracy,
            "detections": detections,
            "pose_label": pose_label,
            "pose_count": len(detections),
            "status": "success" if accuracy > 0 else "no_pose_detected"
        })
    
    except Exception as e:
        print(f"Error in predict_json: {str(e)}")
        return jsonify({
            "error": str(e),
            "accuracy": 0,
            "detections": [],
            "pose_count": 0,
            "status": "error"
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)