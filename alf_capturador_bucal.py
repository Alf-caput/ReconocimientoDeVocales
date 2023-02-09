import cv2
import numpy as np


def main():
    # Classifiers that use algorythm Haar Cascade
    face_detector = cv2.CascadeClassifier(r'haar_tools/haarcascade_frontalface_default.xml')
    mouth_detector = cv2.CascadeClassifier(r'haar_tools/mouth.xml')

    # Video source
    webcam = cv2.VideoCapture(r'videos/vocales0003.mp4')
    while webcam.isOpened() and cv2.waitKey(1) not in (ord('s'), ord('S')):  # While 's' or 'S' not pressed
        # Webcam success on reading, webcam capture
        read_successfully, main_frame = webcam.read()

        if read_successfully:
            # For fewer false positives we preprocess the frame
            processed_frame = cv2.cvtColor(main_frame, cv2.COLOR_BGR2GRAY)
            processed_frame = cv2.blur(processed_frame, (10, 10))

            # Faces inside processed_frame
            faces = face_detector.detectMultiScale(processed_frame, minNeighbors=5)

            # faces is a np.array of np.arrays with x, y, w, h of each face found inside processed_frame
            for x1, y1, w1, h1 in faces:
                # Slices for readability
                face_ypx, face_xpx = slice(y1, y1 + h1 + h1 // 5), slice(x1, x1 + w1)  # We add 1 // 5
                y_third_ypx, x_xpx = slice(2 * h1 // 3, h1), slice(0, w1)  # Not exactly 1 // 3

                # Face rectangle
                cv2.rectangle(main_frame,
                              (x1, y1), (x1 + w1, y1 + h1),
                              (255, 0, 0), 4)

                # Line 2 // 3 face (start point to check for mouth)
                cv2.line(main_frame[face_ypx, face_xpx],
                         (0, 2 * h1 // 3), (w1, 2 * h1 // 3),
                         (0, 255, 0), 4)

                # Mouths found in last "1 // 3" of face
                mouths = mouth_detector.detectMultiScale(main_frame[face_ypx, face_xpx][y_third_ypx, x_xpx])

                if np.any(mouths):
                    # First mouth found inside current face
                    x2, y2, w2, h2 = mouths[0]

                    # Slices for readability
                    mouth_ypx, mouth_xpx = slice(y2, y2 + h2), slice(x2, x2 + w2)

                    # Output frame
                    mouth_frame = main_frame[face_ypx, face_xpx][y_third_ypx, x_xpx][mouth_ypx, mouth_xpx]
                    mouth_frame = cv2.cvtColor(mouth_frame, cv2.COLOR_BGR2GRAY)
                    mouth_frame = cv2.resize(mouth_frame, (400, 200), interpolation=cv2.INTER_AREA)

                    # Mouth rectangle
                    cv2.rectangle(main_frame[face_ypx, face_xpx][y_third_ypx, x_xpx],
                                  (x2, y2), (x2 + w2, y2 + h2),
                                  (0, 0, 255), 4)

                    cv2.imshow('Boca', mouth_frame)

            cv2.imshow('Captura', main_frame)

    webcam.release()
    cv2.destroyAllWindows()
    return


if __name__ == '__main__':
    main()
