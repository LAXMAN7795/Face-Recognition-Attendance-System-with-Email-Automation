# services/face_service.py
import cv2
import face_recognition
import os
from config.settings import IMAGES_DIR
from utils.helpers import parse_image_filename

def load_images(path=IMAGES_DIR):
    images, ids, names = [], [], []

    for file in os.listdir(path):
        img = cv2.imread(os.path.join(path, file))
        if img is not None:
            images.append(img)
            emp_id, name = parse_image_filename(file)
            ids.append(emp_id)
            names.append(name)

    return images, ids, names


def encode_faces(images):
    encodings = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enc = face_recognition.face_encodings(img)
        if enc:
            encodings.append(enc[0])
    return encodings