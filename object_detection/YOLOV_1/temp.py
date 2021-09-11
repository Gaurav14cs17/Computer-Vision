
import joblib, glob, os, cv2
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import svm, metrics
import numpy as np
from sklearn.preprocessing import LabelEncoder
# from CNN_model import CNN_VGG16_Model
from PIL import Image

train_data_set_path = "/content/drive/MyDrive/cnn/ttrain/"
test_dataset_path = "/content/drive/MyDrive/cnn/ttest/4.Apple___healthy/"
svm_model_weight_path = "/content/drive/MyDrive/cnn/"
model_path = os.path.join(svm_model_weight_path ,'models.dat')

import time

print(os.listdir(test_dataset_path))


class Test_our_model:
    def __init__(self):
        self.model_path = model_path
        self.CNN_VGG16_Model_obj = CNN_VGG16_Model()
        self.svm_model = joblib.load(self.model_path)
        self.images_path = test_dataset_path
        self. labels = {"1.Apple___Apple_scab": 0, "2.Apple___Black_rot": 1, "3.Apple___Cedar_apple_rust": 2, "4.Apple___healthy": 3}
        self.total_pred = {}

    def show_reslut(self):
        for filename_1 in os.listdir(test_dataset_path):
            filename_11 = os.path.join(test_dataset_path, filename_1)
            cnt = 0
            file_name = self.labels[filename_1]
            total_count = len(os.listdir(filename_11))
            for filename_2 in os.listdir(filename_11):
                filename = os.path.join(filename_11, filename_2)
                image = Image.open(filename).convert('RGB')
                image_output = self.CNN_VGG16_Model_obj.get_feature(image).data.numpy()
                image_output = image_output.reshape(1, -1)
                pred = self.svm_model.predict(image_output)
                print(pred, filename)
                if pred[0] == file_name:
                    cnt += 1
            self.total_pred[filename_1] = {"Total" :  total_count , "pred" : cnt }
            print(self.total_pred)

        print("total_matrix" , self.total_pred)
        return self.total_pred




if __name__ == '__main__':
    obj = Test_our_model()
    obj.show_reslut()