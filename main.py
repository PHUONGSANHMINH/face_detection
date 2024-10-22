#Detech face of images
import cv2
from cv2.data import haarcascades

image_path = 'imgs/peo1.jpg'



face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

img = cv2.imread(image_path)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)




while True:


    faces = face_detector.detectMultiScale(img_gray, 1.3, 5)
    count = 0
    for (x, y, w, h) in faces:
        img_face = cv2.resize(img[y + 3: y + h - 3, x + 3: x + w - 3], (64,64))
        cv2.imwrite('imgs/peo1_{}.jpg'.format(count), img_face)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        count += 1

    cv2.imshow('FRAME', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()