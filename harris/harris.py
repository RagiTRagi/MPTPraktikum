import cv2
import numpy as np
from scipy import signal


def processImage(frame):


    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray_morm = gray.astype(np.float32) / 255.0
    ix = cv2.Sobel(gray_morm, dx=1, dy=0, ksize=3, ddepth=cv2.CV_32F) / 4
    iy = cv2.Sobel(gray_morm, dx=0, dy=1, ksize=3, ddepth=cv2.CV_32F) / 4

    ix2 = ix**2
    iy2 = iy**2
    ixiy = ix * iy

    N = 7
    bk = np.ones((N,N))
    ix2 = signal.convolve2d(ix2, bk) / (N**2)
    iy2 = signal.convolve2d(iy2, bk) / (N**2)
    ixiy = signal.convolve2d(ixiy, bk) / (N**2)

    kappa = 0.04
    det = ix2 * iy2 - ixiy**2
    trace = ix2 + iy2 
    strength = det - kappa * trace**2

    strength = strength / np.max(strength) 

    bin = np.zeros_like(strength)
    bin[strength > 0.1] = 1.0

    cv2.imshow("Harris Corner Strength", strength)
    cv2.imshow("Harris Corners", bin)

def mainLoop():
    """
    The main loop of this program
    """
    cam = cv2.VideoCapture(0)

    while True:
        _, frame = cam.read()

        processImage(frame)

        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) == 27:
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    mainLoop()
