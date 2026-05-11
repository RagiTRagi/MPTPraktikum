import cv2
import numpy as np


# One shared window name makes it easy to find the camera output again.
WINDOW_NAME = "KLT Optical Flow"

# If fewer than this many points are left, we detect a fresh set of corners.
# This keeps the demo useful even when the tracked object leaves the image.
MIN_TRACKS = 15

# Parameters for cv2.goodFeaturesToTrack. KLT works best on image patches that
# contain visible structure in both directions, therefore we track corners.
FEATURE_PARAMS = dict(
    # Track at most this many points. A small number keeps the visualization
    # readable and makes the algorithm fast enough for a webcam demo.
    maxCorners=100,
    # Keep points whose corner quality is at least 1 percent of the best
    # detected corner. Raising this value keeps only very strong corners.
    qualityLevel=0.01,
    # Keep detected points separated by at least this many pixels. Without this,
    # many points can cluster on the same textured area.
    minDistance=10,
    # Size of the local neighborhood used for the corner score.
    blockSize=7,
)

# Parameters for cv2.calcOpticalFlowPyrLK. OpenCV implements the pyramidal
# Lucas-Kanade tracker, which can follow motions larger than a few pixels.
LK_PARAMS = dict(
    # The tracker compares a small window around each point. A larger window is
    # more stable, but also assumes the same motion over a larger image region.
    winSize=(21, 21),
    # Number of pyramid levels. 0 means only the original image. 3 means the
    # original image plus three smaller versions.
    maxLevel=3,
    # Stop after at most 30 iterations, or earlier when the point update is
    # smaller than epsilon. This is the standard OpenCV stopping criterion.
    criteria=(
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        30,
        0.01,
    ),
)

# Keyboard controls used in the main loop.
RESET_KEY = ord("r")
QUIT_KEYS = {27, ord("q")}  # 27 is the ESC key.


def toGray(frame):
    """
    Convert one camera frame to a grayscale image.

    The camera frame delivered by OpenCV has three color channels in BGR order.
    Both goodFeaturesToTrack and calcOpticalFlowPyrLK operate on single-channel
    images, therefore we use a grayscale version for the actual computation.

    Args:
        frame (np.ndarray): BGR camera image.

    Returns:
        np.ndarray: Grayscale image with the same width and height.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def detectFeaturePoints(gray):
    """
    Detect good points that can be tracked by KLT.

    We use Shi-Tomasi corners via cv2.goodFeaturesToTrack. These are exactly
    the kind of points where the 2x2 KLT equation system is well-conditioned:
    the image patch changes clearly in both x- and y-direction.

    Args:
        gray (np.ndarray): Current grayscale image.

    Returns:
        np.ndarray | None: Points with shape (N, 1, 2), or None if no useful
        corners were found.
    """
    return cv2.goodFeaturesToTrack(gray, **FEATURE_PARAMS)


def trackFeaturePoints(previousGray, gray, previousPoints):
    """
    Track points from the previous frame into the current frame.

    cv2.calcOpticalFlowPyrLK receives the previous image, the current image and
    the points in the previous image. For each point it estimates where the same
    image patch moved in the current image.

    Args:
        previousGray (np.ndarray): Grayscale image from the last time step.
        gray (np.ndarray): Grayscale image from the current time step.
        previousPoints (np.ndarray | None): Points from the previous image with
        shape (N, 1, 2).

    Returns:
        tuple: (currentPoints, status, error) as returned by OpenCV. If there
        are no points to track, all three entries are None.
    """
    if previousPoints is None or len(previousPoints) == 0:
        return None, None, None

    return cv2.calcOpticalFlowPyrLK(
        previousGray,
        gray,
        previousPoints,
        None,
        **LK_PARAMS,
    )


def keepGoodTracks(previousPoints, currentPoints, status):
    """
    Keep only tracks that OpenCV marked as successful.

    The Lucas-Kanade tracker returns a status value for every input point.
    status == 1 means a plausible corresponding point was found. status == 0
    means tracking failed, for example because the point left the image or the
    local patch no longer matches.

    Args:
        previousPoints (np.ndarray): Points in the previous image.
        currentPoints (np.ndarray | None): Tracked points in the current image.
        status (np.ndarray | None): Success flag for every point.

    Returns:
        tuple[np.ndarray, np.ndarray]: Matching arrays with shape (M, 2):
        successful previous points and their successful current positions.
    """
    if currentPoints is None or status is None:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty

    good = status.reshape(-1).astype(bool)
    goodPrevious = previousPoints[good].reshape(-1, 2)
    goodCurrent = currentPoints[good].reshape(-1, 2)

    return goodPrevious, goodCurrent


def drawTracks(frame, tracks, previousPoints, currentPoints):
    """
    Draw the optical flow vectors on top of the live camera image.

    The variable tracks is a persistent image. Every new motion segment is drawn
    into this image, so older paths remain visible. The current feature
    positions are drawn onto a copy of the current camera frame.

    Args:
        frame (np.ndarray): Current BGR camera image.
        tracks (np.ndarray): Persistent BGR image containing old line segments.
        previousPoints (np.ndarray): Successful point locations in the previous
        frame with shape (M, 2).
        currentPoints (np.ndarray): Successful point locations in the current
        frame with shape (M, 2).

    Returns:
        np.ndarray: Camera image with accumulated flow lines and current points.
    """
    overlay = frame.copy()

    for previousPoint, currentPoint in zip(previousPoints, currentPoints):
        oldX, oldY = previousPoint.ravel()
        newX, newY = currentPoint.ravel()

        oldPosition = (int(oldX), int(oldY))
        newPosition = (int(newX), int(newY))

        # Green lines show where a point came from.
        cv2.line(tracks, oldPosition, newPosition, (0, 255, 0), 2)

        # Red circles show the current point positions.
        cv2.circle(overlay, newPosition, 4, (0, 0, 255), -1)

    # Add the persistent track image to the current frame.
    return cv2.add(overlay, tracks)


def drawStatus(frame, trackCount):
    """
    Draw a small status text into the output image.

    This is not part of the algorithm. It only helps during experimentation:
    students can immediately see how many points are still being tracked.

    Args:
        frame (np.ndarray): Output image that will be modified in-place.
        trackCount (int): Number of currently successful tracks.
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

    # Read one frame before the loop. KLT always compares two neighboring
    # images, therefore we need an initial "previous" image.
    ret, frame = cam.read()
    if not ret:
        print("Cannot read first camera image")
        cam.release()
        return

    previousGray = toGray(frame)
    previousPoints = detectFeaturePoints(previousGray)

    # The track image has the same size as the camera frame. It starts black and
    # stores the accumulated green motion lines.
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

        # Redetect points when the user asks for it or when too few tracks are
        # left. We also clear the path image so old and new tracks do not mix.
        if key == RESET_KEY or len(goodCurrent) < MIN_TRACKS:
            tracks = np.zeros_like(frame)
            previousGray = gray
            previousPoints = detectFeaturePoints(previousGray)
            continue

        # The current frame becomes the previous frame for the next iteration.
        # OpenCV expects the points again in shape (N, 1, 2).
        previousGray = gray.copy()
        previousPoints = goodCurrent.reshape(-1, 1, 2)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    mainLoop()
