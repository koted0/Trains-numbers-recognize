import cv2
import numpy as np

net = cv2.dnn.readNet('yolov3_training_last.weights', 'yolov3_testing.cfg')

font = cv2.FONT_HERSHEY_SIMPLEX
colors = np.random.uniform(0, 255, size=(100, 3))

while True:
    # neural part
    img = cv2.imread("example.jpg")
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
    # visual part
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, "number" + " " + confidence, (x, y - 10), font, 1, (0, 0, 0), 2)

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key == 27:  # esc
        break

cv2.destroyAllWindows()
