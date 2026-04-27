import cv2
import numpy as np


def processImage(frame):
    """
    Process the provided image (3-channel BGR) and calculate
    gradients in X and Y direction as well as the gradient magnitude.

    gx and gy shall contain the gradient direction image with values between -1 and +1
    grad shall contain the gradient magnitude image with values between 0 and 1

    :param frame: 3-channel BGR image (np.array)
    :return: 3-tupel (gx, gy, grad) containing the gradient image in X and Y direction as well as the gradient magnitude image (1-channel np.float32 images each).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray_morm = gray.astype(np.float32) / 255.0

    gx = cv2.Sobel(gray_morm, dx=1, dy=0, ksize=3, ddepth=cv2.CV_32F) / 4
    gy = cv2.Sobel(gray_morm, dx=0, dy=1, ksize=3, ddepth=cv2.CV_32F) / 4

    grad = np.sqrt(gx**2 + gy**2) / np.sqrt(2.0)

    return (gx, gy, grad)


def displayImage(gx, gy, grad):
    """
    Apply appropriate scaling and display the provided images.

    :param gx: Gradient image in X-Direction (np.float32 image with values between -1 and +1)
    :param gy: Gradient image in Y-Direction (np.float32 image with values between -1 and +1)
    :param grad: Gradient magnitude image (np.float32 image with values between 0 and 1)
    """
    cv2.imshow("Gradient X", (0.5 * gx + 0.5))
    cv2.imshow("Gradient Y", (0.5 * gy + 0.5))

    cv2.imshow("Gradient Magnitude", 4 * grad)


def mainLoop():
    """
    The main loop of this program
    """
    cap = cv2.VideoCapture(0)  

    while True:
        _, frame = cap.read()

        gx, gy, grad = processImage(frame)

        displayImage(gx, gy, grad)

        cv2.imshow("Kamerabild", frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        

    


if __name__ == "__main__":
    mainLoop()
