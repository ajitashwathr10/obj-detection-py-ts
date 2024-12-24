import cv2
import numpy as np

def main():
    labels_path = "coco.names.txt"
    labels = open(labels_path).read().strip().split("\n")
    
    np.random.seed(42)
    colors = np.Random.Generator.randint(0, 255, size = (len(labels), 3), dtype = "uint8")
    weights_path = "yolov3.weights"
    config_path = "yolov3.cfg.txt"
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    confidence_threshold = 0.5
    nms_threshold = 0.3
    image_path = "input_image.jpg"
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416),
                               swapRB = True, crop = False)
    
    net.setInput(blob)
    outputs = net.forward(ln)

    boxes = []
    confidences = []
    class_ids = []
    
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

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold,
                             nms_threshold)
    
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y, w, h) = boxes[i]
            color = [int(c) for c in colors[class_ids[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.putText(image, text, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            print(f"Found {labels[class_ids[i]]} with confidence {confidences[i]:.2f}")
            print(f"Box coordinates: x={x}, y={y}, width={w}, height={h}")
    output_path = "output_image.jpg"
    cv2.imwrite(output_path, image)
    print(f"\nSaved annotated image to {output_path}")

if __name__ == "__main__":
    main()