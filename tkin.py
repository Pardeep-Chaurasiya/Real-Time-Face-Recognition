import tkinter as tk
import cv2, os
import numpy as np
from PIL import Image
import time

window = tk.Tk()

window.title("Welcome to Facial Recognition System")
message = tk.Label(window, text="Facial Recognition System", bg="Green", fg="white", width=50,
                   height=3, font=('times', 30, 'italic bold underline'))

message.place(x=200, y=20)

lbl = tk.Label(window, text="Enter ID", width=20, height=2, fg="red", bg="yellow", font=('times', 15, ' bold '))
lbl.place(x=400, y=200)

txt = tk.Entry(window, width=20, bg="yellow", fg="red", font=('times', 15, ' bold '))
txt.place(x=700, y=215)

lbl2 = tk.Label(window, text="Enter Name", width=20, fg="red", bg="yellow", height=2, font=('times', 15, ' bold '))
lbl2.place(x=400, y=300)

txt2 = tk.Entry(window, width=20, bg="yellow", fg="red", font=('times', 15, ' bold '))
txt2.place(x=700, y=315)

lbl3 = tk.Label(window, text="Notification : ", width=20, fg="red", bg="yellow", height=2,
                font=('times', 15, ' bold underline '))
lbl3.place(x=400, y=400)

message = tk.Label(window, text="", bg="yellow", fg="red", width=30, height=2, activebackground="yellow",
                   font=('times', 15, ' bold '))
message.place(x=700, y=400)


def clear():
    txt.delete(0, 'end')


def clear2():
    txt2.delete(0, 'end')


def TakeImages():

    Id = (txt.get())
    name = (txt2.get())
    cam = cv2.VideoCapture(0)
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    sampleNum = 0

    while (True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            sampleNum = sampleNum + 1
            cv2.imwrite("TrainingImage\ " + name + "." + Id + '.' + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
            cv2.imshow('frame', img)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        elif sampleNum > 50:
             break
        res = "Images Saved for ID : " + Id + " Name : " + name
        message.configure(text=res)
    cam.release()
    cv2.destroyAllWindows()

id = 0
names = ['None', 'Pardeep','Anchal']
recognizer = cv2.face_LBPHFaceRecognizer.create()
harcascadePath = "haarcascade_frontalface_default.xml"
detector = cv2.CascadeClassifier(harcascadePath)
def TrainImages():
    faces, Id = getImagesAndLabels(path)
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel\Trainner.yml")
    res = "Image Trained"
    message.configure(text=res)

path = 'TrainingImage'
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids

def TrackImages():

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel\Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            id, conf = recognizer.predict(gray[y:y + h, x:x + w])

            if (conf < 100):
                id = names[id]
            else:
                id = "unknown"
                t = time.time()
                cv2.imwrite("Unknown/" + str(t) + ".jpg", gray[y:y + h, x:x + w])
            cv2.putText(im, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.imshow('camera', im)

        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break

    cam.release()
    cv2.destroyAllWindows()


clearButton = tk.Button(window, text="Clear", command=clear, fg="red", bg="yellow", width=20, height=2,
                        activebackground="Red", font=('times', 15, ' bold '))
clearButton.place(x=950, y=200)
clearButton2 = tk.Button(window, text="Clear", command=clear2, fg="red", bg="yellow", width=20, height=2,
                         activebackground="Red", font=('times', 15, ' bold '))
clearButton2.place(x=950, y=300)
takeImg = tk.Button(window, text="Take Images", command=TakeImages, fg="red", bg="yellow", width=20, height=3,
                    activebackground="Red", font=('times', 15, ' bold '))
takeImg.place(x=200, y=500)
trainImg = tk.Button(window, text="Train Images", command=TrainImages, fg="red", bg="yellow", width=20, height=3,
                     activebackground="Red", font=('times', 15, ' bold '))
trainImg.place(x=500, y=500)
trackImg = tk.Button(window, text="Track Images", command=TrackImages, fg="red", bg="yellow", width=20, height=3,
                     activebackground="Red", font=('times', 15, ' bold '))
trackImg.place(x=800, y=500)
quitWindow = tk.Button(window, text="Quit", command=window.destroy, fg="red", bg="yellow", width=20, height=3,
                       activebackground="Red", font=('times', 15, ' bold '))
quitWindow.place(x=1100, y=500)

window.mainloop()
