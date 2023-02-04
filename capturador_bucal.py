import cv2

face_detector = cv2.CascadeClassifier(r'haar_tools/haarcascade_frontalface_default.xml')

webcam = cv2.VideoCapture(0)

while webcam.isOpened() and cv2.waitKey(1) not in (ord('s'), ord('S')):
    read_successfully, frame = webcam.read()
    ########### Option 1 ###########
    # frame = cv2.blur(frame, (5, 5))
    # frame = cv2.Canny(frame, 50, 100)
    # frame = cv2.bitwise_not(frame)

    ########### Option 2 ###########
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if read_successfully:
        faces = face_detector.detectMultiScale(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 200, 50), 4)

        cv2.imshow('Captura', frame)
    else:
        print('Frame not read')


webcam.release()
cv2.destroyAllWindows()
