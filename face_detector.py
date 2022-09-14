import cv2
from random import randrange

# Load some pre-trained data on face frontals from opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# choose an image to detect faces in
# img = cv2.imread('3.jpg')
webcam = cv2.VideoCapture(0)

while True:
    # read curretnt frame
    successful_frame_read, frame = webcam.read()

    # cover to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (y + w, y + h), (randrange(256), randrange(25), randrange(256)), 10)

    cv2.imshow('Face detected', frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:   #ASCII code for Q=81 q=113
        break

webcam.release()

# detect faces
# face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
#
#
# for (x, y, w, h) in face_coordinates:
#     cv2.rectangle(img, (x, y), (y + w, y + h), (randrange(256), randrange(25), randrange(256)), 10)

# (x, y), (y+w, y+h)
# above expression is from coordinates x,y,w,h
# [114, 85, 203, 203]
# [x,y, w, h]

# print(face_coordinates)


# display image with faces


print("code complete")
