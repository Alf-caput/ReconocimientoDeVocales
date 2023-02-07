import cv2

face_detector = cv2.CascadeClassifier(r'haar_tools/haarcascade_frontalface_default.xml')
mouth_detector = cv2.CascadeClassifier(r'haar_tools/mouth.xml')

webcam = cv2.VideoCapture(0)

while webcam.isOpened() and cv2.waitKey(1) not in (ord('s'), ord('S')):

    read_successfully, frame = webcam.read()

    if read_successfully:
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        processed_frame = cv2.blur(processed_frame, (10, 10))
        faces = face_detector.detectMultiScale(processed_frame, minNeighbors=3)

        for (x1, y1, w1, h1) in faces:
            # From frame, we get section that contains the face
            cv2.rectangle(frame,
                          (x1, y1), (x1 + w1, y1 + h1 + h1 // 5),
                          (255, 0, 0), 4)
            # From face, we get the lower section (more likely to find the mouth)
            cv2.rectangle(frame,
                          (x1, y1 + 2 * h1 // 3), (x1 + w1, y1 + 2 * h1 // 3),
                          (0, 255, 0), 4)
            # We get the location of the mouth inside lower_face
            mouths = mouth_detector.detectMultiScale(frame[y1+2*h1//3: y1+h1+h1//5, x1: x1+w1])

            for (x2, y2, w2, h2) in mouths:

                # From lower_face, we get the mouth
                mouth = frame[y2 + y1 + 2 * h1 // 3: y2 + h2 + y1 + 2 * h1 // 3, x2 + x1: x2 + w2 + x1]
                mouth = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)

                # Resize image
                mouth = cv2.resize(mouth, (400, 200), interpolation=cv2.INTER_AREA)

                cv2.imshow('Boca', mouth)

                cv2.rectangle(frame,
                              (x2 + x1, y2 + y1 + 2 * h1 // 3), (x2 + w2 + x1, y2 + h2 + y1 + 2 * h1 // 3),
                              (0, 0, 255), 4)

                break
        # We show the areas processed
        cv2.imshow('Captura', frame)
    else:
        print('Frame not read')
        break


webcam.release()
cv2.destroyAllWindows()
