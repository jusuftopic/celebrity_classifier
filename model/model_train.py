import os
import joblib
import json
import cv2
import numpy as np
import pandas as pd
import pywt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def get_images_paths():
    paths = {}
    cropped_celebrities_dir = os.scandir('../Dataset/Cropped/')
    for cropped_celebrity_dir in cropped_celebrities_dir:
        celebrity_name = cropped_celebrity_dir.path.split('/')[-1]
        for cropped_celebrity_img in os.scandir(cropped_celebrity_dir):
            img_path = cropped_celebrity_img.path.split("\\")[0] + '/' + cropped_celebrity_img.path.split("\\")[1]
            if celebrity_name in paths:
                paths[celebrity_name].append(img_path)
            else:
                paths[celebrity_name] = [img_path]
    return paths


def add_celebrity_name_indexes(paths):
    count = 0
    name_indexes_dict = {}
    for name in paths.keys():
        name_indexes_dict[name] = count
        count += 1
    return name_indexes_dict


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


images_paths_dict = get_images_paths()
celebrity_name_indexes = add_celebrity_name_indexes(images_paths_dict)

model_x = []
model_y = []

for celebrity_name, img_paths in images_paths_dict.items():
    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        wavelet_img = wavelet_transform(img, 'db1', 5)

        resized_img = cv2.resize(img, (32, 32))
        resized_wavelet_img = cv2.resize(wavelet_img, (32, 32))

        img_combined = np.concatenate((resized_img.flatten(), resized_wavelet_img.flatten()))
        model_x.append(img_combined)
        model_y.append(celebrity_name_indexes[celebrity_name])

# Celebrity Classifier training
model_x_train, model_x_test, model_y_train, model_y_test = train_test_split(model_x, model_y, random_state=0)

# SVM
pipeline = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel='rbf', C=10))])
pipeline.fit(model_x_train, model_y_train)
print(classification_report(model_y_test, pipeline.predict(model_x_test)))

# GridSearch CV
model_properties = {
    'logistic_regression': {
        'model': LogisticRegression(solver='liblinear', multi_class='auto'),
        'params': {
            'logisticregression__C': [1, 5, 10]
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'randomforestclassifier__n_estimators': [1, 5, 10]
        }
    },
    'svm': {
        'model': svm.SVC(gamma='auto', probability=True),
        'params': {
            'svc__C': [1, 10, 100, 1000],
            'svc__kernel': ['rbf', 'linear']
        }
    }
}

model_scores = []
best_estimators = {}

for algorithm, params in model_properties.items():
    pipeline = make_pipeline(StandardScaler(), params['model'])
    gs_cv = GridSearchCV(pipeline, params['params'], cv=5, return_train_score=False)
    gs_cv.fit(model_x_train, model_y_train)
    model_scores.append({
        'model': algorithm,
        'best_score': gs_cv.best_score_,
        'best_params': gs_cv.best_params_
    })
    best_estimators[algorithm] = gs_cv.best_estimator_

data_frame = pd.DataFrame(model_scores, columns=['model', 'best_score', 'best_params'])
print(data_frame)

print('SVM score: ' + str(best_estimators['svm'].score(model_x_test, model_y_test)))
print('Logistic regression score: ' + str(best_estimators['logistic_regression'].score(model_x_test, model_y_test)))
print('Random forest score: ' + str(best_estimators['random_forest'].score(model_x_test, model_y_test)))

# SVM performs best, therefore this model will be saved
best_alg = best_estimators['svm']
joblib.dump(best_alg, 'model.pkl')

with open('classifier.json', 'w') as f:
    json.dump(celebrity_name_indexes, f)