import cv2
import os
import shutil

# opencv cascades for detection different part of images... in this case, face and eye
opencv_face = cv2.CascadeClassifier('../opencv/haarcascades/haarcascade_frontalface_default.xml')
opencv_eye = cv2.CascadeClassifier('../opencv/haarcascades/haarcascade_eye.xml')


def create_cropped_folder(cropped_path):
    # remove if cropped folder exists
    if os.path.exists(cropped_path):
        shutil.rmtree(cropped_path)
    os.mkdir(cropped_path)


def get_all_images_paths():
    img_dirs = []
    for img_dir in os.scandir('../Dataset/'):
        if img_dir.is_dir():
            img_dirs.append(img_dir.path)
    return img_dirs


# get cropped images only if both eyes are visible on the picture
def get_cropped_image(path):
    image = cv2.imread(path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    image_faces = opencv_face.detectMultiScale(image_gray, 1.1, 1)
    for(x, y, w, h) in image_faces:
        # Using grayscale images makes the face and eye detection process faster,
        # more reliable and robust
        face_gray = image_gray[y: y+h, x: x+w]
        eyes = opencv_eye.detectMultiScale(face_gray)
        if len(eyes) >= 2:
            print(path)
            # if both eyes are detected, return color image, else null
            return image[y: y+h, x: x+w]


def fill_cropped_folder(cropped_root_path, img_paths):
    for img_path in img_paths:
        celebrity_name = img_path.split('/')[-1]
        img_counter = 1

        for img_dir in os.scandir(img_path):
            cropped_img = get_cropped_image(img_dir.path)
            if cropped_img is not None:
                cropped_img_dir = cropped_root_path + celebrity_name
                if not os.path.exists(cropped_img_dir):
                    os.makedirs(cropped_img_dir)
                    print("Generate cropped images in folder ", cropped_img_dir)
                cropped_img_name = celebrity_name + str(img_counter) + '.jpg'
                cropped_img_path = cropped_img_dir + '/' + cropped_img_name

                cv2.imwrite(cropped_img_path, cropped_img)
                img_counter += 1


cropped_root_path = '../Dataset/Cropped/'

create_cropped_folder(cropped_root_path)
img_paths = get_all_images_paths()
fill_cropped_folder(cropped_root_path, img_paths)