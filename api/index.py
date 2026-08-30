from flask import Flask, request, render_template, send_file, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__, template_folder="../templates")

# Load model - optimize for serverless
model = None

def get_model():
    global model
    if model is None:
        model_path = os.path.join(os.path.dirname(__file__), "../best.pt")
        model = YOLO(model_path)
    return model


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
        
        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")

        model = get_model()
        results = model(img, conf=0.3)
        
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
        
        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Run detection
        model = get_model()
        results = model(img, conf=0.3)
        
        if not results or len(results) == 0:
            return jsonify({
                "accuracy": 0,
                "detections": [],
                "pose_count": 0,
                "status": "no_detection"
            })
        
        result = results[0]
        
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
                    "label": label,
                    "confidence": round(conf, 2),
                    "box": box.xyxy[0].tolist()
                })
        
        # Calculate average accuracy
        if detections:
            avg_accuracy = sum([d["confidence"] for d in detections]) / len(detections)
        else:
            avg_accuracy = 0
        
        return jsonify({
            "accuracy": round(avg_accuracy, 2),
            "detections": detections,
            "pose_count": len(detections),
            "status": "success"
        })
    except Exception as e:
        print(f"Error in predict_json: {str(e)}")
        return jsonify({"error": str(e), "accuracy": 0, "detections": []}), 500


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
