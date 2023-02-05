import cv2

face_detector = cv2.CascadeClassifier(r'haar_tools/haarcascade_frontalface_default.xml')
mouth_detector = cv2.CascadeClassifier(r'haar_tools/mouth.xml')

webcam = cv2.VideoCapture(0)

while webcam.isOpened() and cv2.waitKey(1) not in (ord('s'), ord('S')):

    read_successfully, frame = webcam.read()

    if read_successfully:
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        processed_frame = cv2.blur(processed_frame, (10, 10))
        faces = face_detector.detectMultiScale(processed_frame, minNeighbors=2)

        for (x1, y1, w1, h1) in faces:

            cv2.rectangle(frame,
                          (x1, y1), (x1 + w1, y1 + h1 + h1 // 8),
                          (255, 0, 0), 4)
            cv2.rectangle(frame,
                          (x1, y1 + 2 * h1 // 3), (x1 + w1, y1 + 2 * h1 // 3),
                          (0, 255, 0), 4)
            mouths = mouth_detector.detectMultiScale(frame[y1+2*h1//3: y1+h1+h1//10, x1: x1+w1])

            for (x2, y2, w2, h2) in mouths:
                cv2.rectangle(frame[y1 + 2 * h1 // 3: y1 + h1 + h1 // 10, x1: x1 + w1],
                              (x2, y2), (x2 + w2, y2 + h2),
                              (0, 0, 255), 4)
                break

        cv2.imshow('Captura', frame)
    else:
        print('Frame not read')
        break


webcam.release()
cv2.destroyAllWindows()
