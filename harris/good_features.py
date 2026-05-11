import cv2
import numpy as np


def processImage(frame):
    overlay = frame.copy()

    # TODO: Convert the image to gray

    # TODO: Use cv2.goodFeaturesToTrack to detect good corners

    # TODO: Draw the detected corners on top of the camera image

    # TODO: Return the camera image with the corner overlay
    return overlay


def mainLoop():
    """
    The main loop of this program
    """
    # TODO: Open the default camera
    cam = cv2.VideoCapture(0)

    while True:
        # TODO: Read next image from camera
        _, frame = cam.read()

        # TODO: Detect good features and draw them on the camera image
        overlay = processImage(frame)

        # TODO: Display the camera image with the detected features
        cv2.imshow("Good Features To Track", overlay)

        # TODO: Break the infinite loop when the users presses ESCAPE (27)
        if cv2.waitKey(1) == 27:
            break

    # TODO: Release the capture and writer objects
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    mainLoop()
