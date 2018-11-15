import argparse

import cv2
import numpy as np

import util_box

parser = argparse.ArgumentParser()
parser.add_argument('--image', required=True,
                    help='path to input image')
parser.add_argument('--config', required=True,
                    help='path to yolo config file')
parser.add_argument('--weights', required=True,
                    help='path to yolo pre-trained weights')
parser.add_argument('--classes', required=True,
                    help='path to text file containing class names')
parser.add_argument('--scale', default=0.00392, type=int)
parser.add_argument('--width', default=416, type=int)
parser.add_argument('--height', default=416, type=int)
args = parser.parse_args()


def get_output_layers(input_net):
    layer_names = input_net.getLayerNames()

    output_layers = [layer_names[index[0] - 1] for index in input_net.getUnconnectedOutLayers()]

    return output_layers


if __name__ == '__main__':
    cap = cv2.VideoCapture(args.image)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    image_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    image_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('out.avi', fourcc, fps, (image_width, image_height))
    scale = args.scale
    with open(args.classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    net = cv2.dnn.readNet(args.weights, args.config)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        blob = cv2.dnn.blobFromImage(frame, scale, (args.width, args.height), (0, 0, 0), True, crop=False)

        net.setInput(blob)

        outs = net.forward(get_output_layers(net))

        class_ids: list = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * image_width)
                    center_y = int(detection[1] * image_height)
                    w = int(detection[2] * image_width)
                    h = int(detection[3] * image_height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        for i in indices:
            i = i[0]
            if str(classes[class_ids[i]]) == 'person':
                box = boxes[i]
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]
                util_box.Box(x, y, w, h)
