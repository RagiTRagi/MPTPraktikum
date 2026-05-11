import cv2
import numpy as np


WINDOW_NAME = "KLT Optical Flow"
MIN_TRACKS = 15

# These parameters are already chosen for this exercise. You can experiment
# with them later, but they are not the important implementation step here.
FEATURE_PARAMS = dict(
    maxCorners=100,
    qualityLevel=0.01,
    minDistance=10,
    blockSize=7,
)

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        30,
        0.01,
    ),
)

RESET_KEY = ord("r")
QUIT_KEYS = {27, ord("q")}


def toGray(frame):
    """
    Convert one BGR camera frame to a grayscale image.

    Input:
        frame: One color image from the webcam. OpenCV stores color images in
        BGR channel order, so the shape is usually (height, width, 3).

    Expected output:
        A single-channel grayscale image with shape (height, width). Return the
        image itself, do not show it in this function.
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return gray


def detectFeaturePoints(gray):
    """
    Detect points that should be tracked by KLT.

    Input:
        gray: The current grayscale image returned by toGray.

    Expected output:
        Either None, if no useful points were found, or an OpenCV point array
        with shape (N, 1, 2). The last dimension contains x and y coordinates.
    """
    corners = cv2.goodFeaturesToTrack(gray, **FEATURE_PARAMS)

    return corners


def trackFeaturePoints(previousGray, gray, previousPoints):
    """
    Track points from the previous image into the current image.

    Inputs:
        previousGray: The grayscale camera image from the previous time step.
        gray: The grayscale camera image from the current time step.
        previousPoints: Points detected or tracked in previousGray. The shape
        is (N, 1, 2), unless there are currently no points.

    Expected output:
        The three return values of cv2.calcOpticalFlowPyrLK:
        currentPoints, status and error. If there are no previous points,
        return None, None, None.
    """
    if previousPoints is None or len(previousPoints) == 0:
        return None, None, None

    current, status, error =  cv2.calcOpticalFlowPyrLK(previousGray, gray, previousPoints, None, **LK_PARAMS)

    return current, status, error


def keepGoodTracks(previousPoints, currentPoints, status):
    """
    Keep only point tracks where OpenCV returned status == 1.

    Inputs:
        previousPoints: Old point positions in the previous image.
        currentPoints: New point positions estimated by cv2.calcOpticalFlowPyrLK.
        status: One success flag for each point. A value of 1 means the point
        was tracked successfully, 0 means the track failed.

    Expected output:
        Two arrays with shape (M, 2): the successful previous point positions
        and their matching successful current point positions.
    """
    if currentPoints is None:
        return np.empty([0,2], dtype=np.float32), np.empty([0,2], dtype=np.float32)

    good = status.reshape(-1).astype(bool)
    goodPrevious = previousPoints[good].reshape(-1, 2)
    goodCurrent = currentPoints[good].reshape(-1, 2)
   
    return goodPrevious, goodCurrent


def drawTracks(frame, tracks, previousPoints, currentPoints):
    """
    Draw optical flow vectors and current feature positions.
    """
    overlay = frame.copy()

    for previousPoint, currentPoint in zip(previousPoints, currentPoints):
        oldX, oldY = previousPoint.ravel()
        newX, newY = currentPoint.ravel()

        oldPosition = (int(oldX), int(oldY))
        newPosition = (int(newX), int(newY))

        cv2.line(tracks, oldPosition, newPosition, (0, 255, 0), 2)
        cv2.circle(overlay, newPosition, 4, (0, 0, 255), -1)

    return cv2.add(overlay, tracks)


def drawStatus(frame, trackCount):
    """
    Draw a small status text into the output image.
    """
    text = f"tracks: {trackCount}   r: reset   ESC/q: quit"
    cv2.putText(
        frame,
        text,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def mainLoop():
    """
    Open the webcam and run the KLT tracker until the user quits.
    """
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Cannot open camera")
        return

    ret, frame = cam.read()
    if not ret:
        print("Cannot read first camera image")
        cam.release()
        return

    previousGray = toGray(frame)
    previousPoints = detectFeaturePoints(previousGray)
    tracks = np.zeros_like(frame)

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = toGray(frame)

        currentPoints, status, _ = trackFeaturePoints(
            previousGray,
            gray,
            previousPoints,
        )
        goodPrevious, goodCurrent = keepGoodTracks(
            previousPoints,
            currentPoints,
            status,
        )

        output = drawTracks(frame, tracks, goodPrevious, goodCurrent)
        drawStatus(output, len(goodCurrent))
        cv2.imshow(WINDOW_NAME, output)

        key = cv2.waitKey(1) & 0xFF
        if key in QUIT_KEYS:
            break

        if key == RESET_KEY or len(goodCurrent) < MIN_TRACKS:
            tracks = np.zeros_like(frame)
            previousGray = gray
            previousPoints = detectFeaturePoints(previousGray)
            continue

        previousGray = gray.copy()
        previousPoints = goodCurrent.reshape(-1, 1, 2)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    mainLoop()
