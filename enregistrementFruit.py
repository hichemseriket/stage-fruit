import cv2
import operator
import common as c


# jai créé un xml pour banane pour essayé au moins de lancé le script de la camera et voir si elle sais reconnaitre une banane avec le label que jai créé sur labelimg
fruit_cascade=cv2.CascadeClassifier("hichem_fruit.xml")
cap=cv2.VideoCapture(0)

id=0
while True:
    ret, frame=cap.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face=fruit_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(c.min_size, c.min_size))
    for x, y, w, h in face:
        cv2.imwrite("non-classees/p-{:d}.png".format(id), frame[y:y+h, x:x+w])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        id+=1
    cv2.imshow('video', frame)
    key=cv2.waitKey(1)&0xFF
    if key==ord('q'):
        break
    if key==ord('a'):
        for cpt in range(100):
            ret, frame=cap.read()

cap.release()
cv2.destroyAllWindows()
