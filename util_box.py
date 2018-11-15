import numpy as np


class TraceBox:
    def __init__(self):
        self.boxes = []


class Box:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def __call__(self, boxes):
        for box in boxes:
            np.sqrt((abs(self.x - box.x)) ** 2 + (abs(self.y - box.y)) ** 2)

    def iou(self, box):
        return self.x <= box.x <= self.x + self.width and self.y <= box.y <= self.y + self.height
