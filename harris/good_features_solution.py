import cv2
import numpy as np


def processImage(frame):
    # GoodFeaturesToTrack works on a single channel image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect strong, well separated corners
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=100,
        qualityLevel=0.01,
        minDistance=10,
    )

    # Draw the detected corners on top of the camera image
    overlay = frame.copy()
    if corners is not None:
        corners = np.int32(corners)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(overlay, (x, y), 4, (0, 255, 0), -1)

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
