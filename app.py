from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok = True)
os.makedirs(OUTPUT_FOLDER, exist_ok = True)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_model():
    weights_path = "yolov3.weights"
    config_path = "yolov3.cfg"
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    labels_path = "coco.names"
    labels = open(labels_path).read().strip().split("\n")
    
    np.random.seed(42)
    colors = np.random.Generator(0, 255, size = (len(labels), 3), dtype = "uint8")
    
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    
    return net, labels, colors, ln

def detect_objects(image_path, net, labels, colors, ln):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB = True, crop = False)
    net.setInput(blob)
    outputs = net.forward(ln)
    
    confidence_threshold = 0.5
    nms_threshold = 0.3
    
    boxes = []
    confidences = []
    class_ids = []
    detections = []
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > confidence_threshold:
                box = detection[0:4] * np.array([width, height, width, height])
                (center_x, center_y, w, h) = box.astype("int")
                
                x = int(center_x - (w / 2))
                y = int(center_y - (h / 2))
                
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y, w, h) = boxes[i]
            color = [int(c) for c in colors[class_ids[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            detections.append({
                "label": labels[class_ids[i]],
                "confidence": float(confidences[i]),
                "box": {"x": x, "y": y, "width": w, "height": h}
            })
    
    return detections, image
net, labels, colors, ln = initialize_model()

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)
        
        try:
            detections, annotated_image = detect_objects(input_path, net, labels, colors, ln)
            output_filename = f"detected_{filename}"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            cv2.imwrite(output_path, annotated_image)
            return jsonify({
                "detections": detections,
                "annotated_image_url": f"/output/{output_filename}"
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        
        finally:
            if os.path.exists(input_path):
                os.remove(input_path)
    
    return jsonify({"error": "Invalid file type"}), 400

@app.route('/output/<filename>')
def output_file(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename))

if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0', port = 5000)