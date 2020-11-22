import cv2
from random import randrange
# loading some pre trained data on face frontals from opencv
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Choosing an image to detect faces in
# here you can add your photo for detection
img = cv2.imread("MYPHOTO.jpeg")

# Converting it into greyscale
grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Detecting faces by putting our grey scale image in the above trained data
face_coordinates = trained_face_data.detectMultiScale(grey_img)
print(face_coordinates)

# Draw rectangle around face
for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),5)

cv2.imshow("This sis Fahads Photo",img)
cv2.waitKey()
print("Code Finished")