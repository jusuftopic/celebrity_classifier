import base64
import json

import cv2
import joblib
import numpy as np
import pywt

celebrity_name_index_dict = {}
index_celebrity_name_dict = {}
model = None


def get_cv2_image_from_base64_string(b64str):
    '''
    credit: https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library
    '''
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def wavelet_transform(image, mode, level):
    imArray = image
    # Datatype conversions
    # convert to grayscale
    imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
    # convert to float
    imArray = np.float32(imArray)
    imArray /= 255
    # compute coefficients
    coeffs = pywt.wavedec2(imArray, mode, level=level)

    # Process Coefficients
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0

    # reconstruction
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)

    return imArray_H


def get_cropped_images(img_path, img_base64):
    cropped_images = []
    opencv_face = cv2.CascadeClassifier('../opencv/haarcascades/haarcascade_frontalface_default.xml')
    opencv_eye = cv2.CascadeClassifier('../opencv/haarcascades/haarcascade_eye.xml')

    image = cv2.imread(img_path) if img_path else get_cv2_image_from_base64_string(img_base64)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    image_faces = opencv_face.detectMultiScale(image_gray, 1.3, 5)

    for(x, y, w, h) in image_faces:
        face = image[y:y+h, x:x+w]
        face_gray = image_gray[y:y+h, x:x+w]
        eyes = opencv_eye.detectMultiScale(face_gray)

        if len(eyes) >= 2:
            cropped_images.append(face)

    return cropped_images


def classify_image(img_base64, img_path=None):
    cropped_images = get_cropped_images(img_path, img_base64)
    result = []

    for image in cropped_images:
        wavelet_image = wavelet_transform(image, 'db1', 5)

        resized_image = cv2.resize(image, (32, 32))
        resized_wavelet_image = cv2.resize(wavelet_image, (32, 32))

        image_combined = np.concatenate((resized_image.flatten(), resized_wavelet_image.flatten()))

        final_image = image_combined.reshape(1, 32*32*3 + 32*32).astype(float)
        result.append({
            'celebrity': get_name_from_index(model.predict(final_image)[0]),
            'probability': np.around(model.predict_proba(final_image)*100, 2).tolist()[0],
            'celebrity_dictionary': celebrity_name_index_dict
        })

    return result


def get_name_from_index(index):
    return index_celebrity_name_dict[index]


def load_classifier():
    global celebrity_name_index_dict
    global index_celebrity_name_dict

    with open('./artifacts/classifier.json', 'r') as f:
        celebrity_name_index_dict = json.load(f)
        # swap dictionary
        index_celebrity_name_dict = {index: name for name, index in celebrity_name_index_dict.items()}


def load_model():
    global model
    if model is None:
        with open('./artifacts/model.pkl', 'rb') as f:
            model = joblib.load(f)


if __name__ == '__main__':
    load_classifier()
    load_model()
    print(classify_image(None, '../Dataset/Cropped/Ronaldo/Ronaldo4.jpg'))