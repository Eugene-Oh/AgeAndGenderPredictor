# Eugene Oh
# TCSS 455 - Machine Learning
# Image Prediciton Project

# Originally my previous project to predict age and gender among a given dataset for a machine learning class.
# Has been modified to predict age and gender of any image given in the "inputimageshere" folder.

# Some imports and other code chunks were used previously on a different project.

import cv2
import sys
import os
import os.path
import numpy as np
# import pandas as pd
from csv import reader
# from lxml import etree as et
# from sklearn.metrics import accuracy_score

CONFIDENCE_LEVEL = .80

SAMPLE_SIZE = 99999

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

AGES = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']

GENDER_LIST = ['Male', 'Female']

# Retrieves the three different neural network models and their weights used for facial, gender, and age recognition.
face_model = cv2.dnn.readNetFromCaffe("imageresources/deploy.prototxt.txt", "imageresources/res10_300x300_ssd_iter_140000_fp16.caffemodel")

gender_net = cv2.dnn.readNetFromCaffe("imageresources/deploy_gender.prototxt", "imageresources/gender_net.caffemodel")

age_net = cv2.dnn.readNetFromCaffe("imageresources/deploy_age.prototxt", "imageresources/age_net.caffemodel")

# Returns four points representing a box around the user's face
def face_detection(image): 
    # Get width and height of image
    height = image.shape[0]
    width = image.shape[1]
    # Pre-processing of image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    # Input the image into the model
    face_model.setInput(blob)
    # Get result from model
    output = np.squeeze(face_model.forward())
    faces = []
    for i in range(0, output.shape[0]):
        confidence = output[i, 2]
        # If confidence level is suitable, then calculate coordinates of face
        if confidence > CONFIDENCE_LEVEL:
            # Retrieve coordinates and upscale them
            box = output[i, 3:7] * np.array([width, height, width, height])
            # Integer conversion
            start_x, start_y, end_x, end_y = box.astype(int)
            start_x, start_y, end_x, end_y = start_x - \
            10, start_y - 10, end_x + 10, end_y + 10
            start_x = 0 if start_x < 0 else start_x
            start_y = 0 if start_y < 0 else start_y
            end_x = 0 if end_x < 0 else end_x
            end_y = 0 if end_y < 0 else end_y
            faces.append((start_x, start_y, end_x, end_y))
    return faces

# Detects gender using facial recognition
def gender_detection(image):
    # Retrieve the faces of each person in the image
    image_copy = image.copy()
    faces = face_detection(image_copy)
    # If no face is detected, set default gender to female.
    gender = "female"
    # Set the right-most person's face to the gender if there are multiple people.
    if len(faces) >= 1:
        for i, (start_x, start_y, end_x, end_y) in enumerate(faces):
            face_img = image[start_y: end_y, start_x: end_x]
            blob = cv2.dnn.blobFromImage(image=face_img, scalefactor=1.0, size=(
            227, 227), mean=MODEL_MEAN_VALUES, swapRB=False, crop=False)
            # Passing the blob to the neural network to get prediction
            # show_image(blob, "image")
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            i = gender_preds[0].argmax()
            gender = GENDER_LIST[i]
            # print(gender)
    return gender

# Detects age using facial recognition
def age_detection(image):
    image_copy = image.copy()
    faces = face_detection(image_copy)
    # The default age
    age = "xx-24"
    # Set the right-most person's face to the age if there are multiple people.
    if len(faces) >= 1:
        for i, (start_x, start_y, end_x, end_y) in enumerate(faces):
            face_img = image[start_y: end_y, start_x: end_x]
            blob = cv2.dnn.blobFromImage(image=face_img, scalefactor=1.0, size=(
            227, 227), mean=MODEL_MEAN_VALUES, swapRB=False)
            # Passing the blob to the neural network to get prediction
            age_net.setInput(blob)
            age_preds = age_net.forward()
            i = age_preds[0].argmax()
            # Converting output range to our required age ranges
            if i <= 4:
                age = "xx-24"
            elif i == 5:
                age = "25-34"
            elif i == 6:
                age = "35-49"
            else:
                age = "50-xx"
            # print(age)
    return age
        
# Shows the user picture
def show_image(image, title):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Creates an xml file for each user in the training data set.
def create_xml(output, outputfile):
    rowcounter = 0
    for userid in output['userid']:
        root = et.Element('user')
        root.set('id', str(userid))
        root.set('age_group', str(output['age'].values[rowcounter]))
        if int(output.loc[rowcounter, 'gender']) == 1:
            root.set('gender', "female")
        else:
            root.set('gender', "male")
        root.set('extrovert', "3.4")
        root.set('neurotic', "2.7")
        root.set('agreeable', "3.5")
        root.set('conscientious', "3.4")
        root.set('open', "3.9")
        tree = et.ElementTree(root)
        rowcounter = rowcounter + 1
        with open(outputfile + str(userid) + '.xml', "wb") as fh:
            tree.write(fh)    

# Tests the age and gender results against the CSV file
def test_data(reader, output):
    # Changes all age ranges in original dataframe to compute accuracy
    rowcounter = 0
    for row in reader.iterrows():
        if int(reader.loc[rowcounter, 'age']) <= 24:
            reader.loc[rowcounter, 'age'] = "xx-24"
        elif int(reader.loc[rowcounter, 'age']) >= 25 and int(reader.loc[rowcounter, 'age']) <= 34:
            reader.loc[rowcounter, 'age'] = "25-34"
        elif int(reader.loc[rowcounter, 'age']) >= 35 and int(reader.loc[rowcounter, 'age']) <= 49:
            reader.loc[rowcounter, 'age'] = "35-49"
        elif int(reader.loc[rowcounter, 'age']) >= 50:
            reader.loc[rowcounter, 'age'] = "50-xx"
        rowcounter = rowcounter + 1
    # print(reader)   
    # print(output)
    # Tests the generated result against the real results.
    y_test = reader['gender']
    y_predicted = output['gender']
    print("Gender Accuracy: %.2f" % accuracy_score(y_test,y_predicted) + " sample size of " + str(SAMPLE_SIZE))
    y_test = reader['age']
    y_predicted = output['age']
    print("Age Accuracy: %.2f" % accuracy_score(y_test,y_predicted) + " sample size of " + str(SAMPLE_SIZE))

# This main method was used for calculating age and gender among users for a given dataset.
def previous_main_for_datasets():
    # Get input and output paths from user
    inputpath = os.path.abspath(os.path.dirname(__file__)) + sys.argv[2]
    outputfile = os.path.abspath(os.path.dirname(__file__)) + sys.argv[4]
    is_training = 0
    # Check if we are either training or testing
    if os.path.isfile(inputpath + "profile/profiletraining.csv"):
        is_training = 1
        reader = pd.read_csv(inputpath + "profile/profiletraining.csv", index_col=0, nrows = SAMPLE_SIZE)
    elif os.path.isfile(inputpath + "profile/profile.csv"):
        reader = pd.read_csv(inputpath + "profile/profile.csv", index_col=0, nrows = SAMPLE_SIZE)
    # Create a dataframe of n size and sets the values other than user ID to 0.
    output = reader.copy()
    for word in list(output.columns.values)[1:]:
        output[word] = 0
    # Retrieves each user, calculates gender and age based on each image
    rowcounter = 0     
    for id in output['userid']:
        image = cv2.imread(inputpath + "image/" + id + ".jpg")
        age = age_detection(image)
        gender = gender_detection(image)
        # show_image(image, id)
        output.loc[rowcounter, 'age'] = age
        # Changes the gender value in the output dataframe to a numerical value.
        if gender == 'Female':
            output.loc[rowcounter, 'gender'] = 1.0
        else:
            output.loc[rowcounter, 'gender'] = 0.0
        rowcounter = rowcounter + 1
    # Checks if we are using the training data or testing data.
    if is_training:
        test_data(reader, output)
    create_xml(output, outputfile)

def main():
    for image_name in os.listdir("inputimageshere"):
        if (image_name.endswith(".png") or image_name.endswith(".jpg") or image_name.endswith(".jpeg")):
            image = cv2.imread("inputimageshere/" + image_name)

            # Resize of image to be displayed, not used for the model.
            if (image.shape[1] > 1000):
                scale_percent = 50
            else:
                scale_percent = 150
            width = int(image.shape[1] * scale_percent / 100)
            height = int(image.shape[0] * scale_percent / 100)
            dim = (width, height)

            # Get predictions by passing the original image through the model.
            face = face_detection(image)
            age = age_detection(image)
            gender = gender_detection(image)
            
            # Displaying the image with its predictions.
            for i, (start_x, start_y, end_x, end_y) in enumerate(face):
                box_color = (255, 0, 0) if gender == "Male" else (147, 20, 255)
                cv2.rectangle(image, (start_x, start_y), (end_x, end_y), box_color, 1)
                label = f"Gender: {gender} Age: {age}"
                yPos = start_y - 15
                while yPos < 15:
                    yPos += 15
                if (scale_percent == 150):
                    cv2.putText(image, label, (start_x, yPos), cv2.FONT_HERSHEY_SIMPLEX, .38, box_color, 1)
                else:
                    cv2.putText(image, label, (start_x, yPos), cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 1)
                resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
                show_image(resized, image_name)

main()
# previous_main_for_datasets()




