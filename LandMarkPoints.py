import csv
import subprocess
from pathlib import Path
import os
import cv2
import dlib
import glob
from imutils import face_utils
import shutil

ROOT_PATH = '/home/bortoletti/projects/mastership/FaceLandmarksExtractor'


class LandMarkPoints:

    def open_face(self, final_image_path, output_folder, save_image=True):
        exec_path = str(Path(ROOT_PATH, "exec/openFace/FeatureExtraction"))
        print('OPEN FACE')
        print('filename: ' + final_image_path)
        subprocess.run(
            [exec_path + " -f " + str(final_image_path) + " -of " +
             output_folder + "/points.csv " + " -no3Dfp " + " -noMparams " +
             " -noPose " + " -noAUs " + " -noGaze "], shell=True)
        points_dict = {}
        output_file = output_folder + "/" + "points.csv"
        points_file = open(output_file, 'r')
        reader = csv.reader(points_file)
        headers = next(reader)
        for row in reader:
            for h, v in zip(headers, row):
                points_dict[h.strip()] = float(v)
        points_file.close()
        # shutil.rmtree(output_folder)
        if save_image:
            img = cv2.imread(final_image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            output_filename = output_folder + '_openFace.jpg'
            self.__save_image(gray, points_dict, output_filename)
        return points_dict

    def open_cv_haar_face_detector(self, final_image_path, output_folder, save_image=True):
        exec_path = str(Path(ROOT_PATH, "exec/haarcascade_frontalface_alt2.xml"))
        img = cv2.imread(final_image_path)
        print('HAAR')
        print('filename: ' + final_image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(exec_path)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        points = None
        if len(faces) == 1:
            points = self.__extract_land_marks(faces[0], gray)
        if save_image:
            output_filename = output_folder + '_openCvHaar.jpg'
            self.__save_image(img, points, output_filename, faces[0])
        return points

    def open_cv_dnn_face_detector(self, final_image_path, output_folder, save_image=True):
        config_file = str(Path(ROOT_PATH, "exec/opencv_face_detector.pbtxt"))
        model = str(Path(ROOT_PATH, "exec/opencv_face_detector_uint8.pb"))
        img = cv2.imread(final_image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width, channels = img.shape
        print('DNN')
        print('filename: ' + final_image_path)
        net = cv2.dnn.readNetFromTensorflow(model, config_file)
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], False, False)
        net.setInput(blob)
        detections = net.forward()
        points = None
        face = None
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.9:  # 0.9 threshold
                x1 = int(detections[0, 0, i, 3] * width)
                y1 = int(detections[0, 0, i, 4] * height)
                x2 = int(detections[0, 0, i, 5] * width)
                y2 = int(detections[0, 0, i, 6] * height)
                w = x2 - x1
                h = y2 - y1
                face = x1, y1, w, h
                points = self.__extract_land_marks(face, gray)
        if save_image:
            output_filename = output_folder + '_openCvDNN.jpg'
            self.__save_image(img, points, output_filename, face)
        return points

    def dlib_hog(self, final_image_path, output_folder, save_image=True):
        img = cv2.imread(final_image_path)
        print('HOG')
        print('filename: ' + final_image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hogFaceDetector = dlib.get_frontal_face_detector()
        faceRects = hogFaceDetector(img, 0)
        points = None
        face = None
        for faceRect in faceRects:
            x1 = faceRect.left()
            y1 = faceRect.top()
            x2 = faceRect.right()
            y2 = faceRect.bottom()
            w = x2 - x1
            h = y2 - y1
            face = x1, y1, w, h
            points = self.__extract_land_marks(face, gray)
        if save_image:
            output_filename = output_folder + '_dlibHOG.jpg'
            print(face)
            self.__save_image(img, points, output_filename, face)
        return points

    def dlib_cnn(self, final_image_path, output_folder, save_image=True):
        face_detector = str(Path(ROOT_PATH, "exec/mmod_human_face_detector.dat"))
        img = cv2.imread(final_image_path)
        scale_percent = 50  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        print('CNN')
        print('filename: ' + final_image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dnnFaceDetector = dlib.cnn_face_detection_model_v1(face_detector)
        faceRects = dnnFaceDetector(img, 0)
        points = None
        face = None
        for faceRect in faceRects:
            x1 = faceRect.rect.left()
            y1 = faceRect.rect.top()
            x2 = faceRect.rect.right()
            y2 = faceRect.rect.bottom()
            w = x2 - x1
            h = y2 - y1
            face = x1, y1, w, h
            points = self.__extract_land_marks(face, gray)
        if save_image:
            output_filename = output_folder + '_dlibCNN.jpg'
            self.__save_image(img, points, output_filename, face)
        return points

    def __save_image(self, gray, points_dict, output_filename, face=None):
        for i in range(0, 68):
            x = int(points_dict['x_' + str(i)])
            y = int(points_dict['y_' + str(i)])
            cv2.circle(gray, (x, y), 7, (255, 255, 0), -1)
            cv2.putText(gray, str(i), (x, y), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.7,
                        color=(0, 0, 0))
        base_path = os.path.dirname(output_filename)
        if not os.path.exists(base_path):
            os.makedirs(base_path, exist_ok=True)
        print(base_path)
        if face is not None:
            x, y, w, h = face
            gray = gray[y - 300:y + h + 300, x - 300:x + w + 300]
        cv2.imwrite(output_filename, gray)

    def __extract_land_marks(self, face, gray):
        shape_predictor = str(Path(ROOT_PATH, "exec/shape_predictor_68_face_landmarks.dat"))
        predictor = dlib.shape_predictor(shape_predictor)
        x, y, w, h = face
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = predictor(gray, dlib_rect)
        shape = face_utils.shape_to_np(shape)
        points = {}
        for (i, (x, y)) in enumerate(shape):
            points['x_' + str(i)] = x
            points['y_' + str(i)] = y
        return points


if __name__ == '__main__':
    teste = LandMarkPoints()
    cropped_images = os.listdir(ROOT_PATH + "/cropped_images")
    for i in cropped_images:
        try:
            src = ROOT_PATH + "/cropped_images/" + i
            # teste.dlib_hog(src, ROOT_PATH + "/points_in_face/" + i.split('_')[0])
            # teste.open_face(src, ROOT_PATH + "/points_in_face/" + i.split('_')[0])
            teste.open_cv_dnn_face_detector(src, ROOT_PATH + "/points_in_face/" + i.split('_')[0])
            teste.open_cv_haar_face_detector(src, ROOT_PATH + "/points_in_face/" + i.split('_')[0])
        except:
            print("")
    # print(teste.dlib_hog('/home/bortoletti/Downloads/eu.jpg', ''))
    # print(teste.dlib_hog('/home/bortoletti/projects/mastership/oxford_images/22q11/image_10001_jpg.jpg', ''))
    # print(teste.open_cv_dnn_face_detector('/home/bortoletti/projects/mastership/oxford_images/22q11/image_10001_jpg.jpg', ''))
    # print(teste.open_cv_haar_face_detector('/home/bortoletti/projects/mastership/oxford_images/22q11/image_10001_jpg.jpg', ''))
    # print(teste.open_face('/home/bortoletti/projects/mastership/oxford_images/22q11/image_10001_jpg.jpg',teste.ROOT_PATH))
