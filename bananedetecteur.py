import cv2
import numpy as np

# putain d xml a la con il me bloque le generateur que jai cest de la merde de la grosse merde
# object_cascade=cv2.CascadeClassifier("./hichem_fruit.xml")
# object_cascade = cv2.CascadeClassifier("./cars.xml")
# j'ai mis un xml depuis internet mais qui ne corespond pas a mes dossier va falloir que je le fasse correspondre au nombre label et retiré les prix'

# object_cascade = cv2.CascadeClassifier("/fruitxmltest.xml")
object_cascade = cv2.CascadeClassifier("/samplefruit.xml")

# ici j'utilise ma camera'
cap = cv2.VideoCapture(0)

# test avec video que je vien de telechargé elle est nul mais hope il va reconnaitre une banane
# cap=cv2.VideoCapture("banane.mp4")

while True:
    ret, frame = cap.read()
    tickmark = cv2.getTickCount()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    object = object_cascade.detectMultiScale(gray, scaleFactor=1.10, minNeighbors= 3)
    for x, y, w, h in object:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - tickmark)
    cv2.putText(frame, "FPS: {:05.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.imshow('video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
