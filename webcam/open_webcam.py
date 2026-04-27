import cv2
import numpy as np

def drawHistogram(gray):
    hist, _ = np.histogram(gray, bins=range(256))
    hist = np.float32(hist)
    hist = hist / np.max(hist)

    H = 128
    histImage = np.zeros((H, 256))
    for bin, value in enumerate(hist):
        x0, y0 = bin, H
        x1, y1 = bin, int(H * (1.0 - value))
        cv2.line(histImage, (x0, y0), (x1, y1), bin / 256.0, 1)

    return histImage

def adjustBrightness(gray, brightness, contrast):
    mean, std = np.mean(gray), np.std(gray)

    norm = brightness + contrast * (gray - mean) / std
    return np.uint8(np.clip(norm, 0.0, 1.0) * 255.0)

cap = cv2.VideoCapture(0)  
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        exit()

    cv2.imshow("Kamerabild", frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    cv2.imshow("Graubild", gray)
    cv2.imshow("Graubild - Histogramm", drawHistogram(gray))

    norm = adjustBrightness(gray, brightness=0.5, contrast=1.0)
    cv2.imshow("Normalisiert", norm)
    cv2.imshow("Normalisiert - Histogramm", drawHistogram(norm))

    key = cv2.waitKey(1)
    if key == ord("+"):
        contrast = np.clip(contrast + 0.01, 0.0, 1.0)
    
    if key == ord("-"):
        contrast = np.clip(contrast - 0.01, 0.0, 1.0)

    if key == ord("1"):
        brightness = np.clip(brightness - 0.05, 0.0, 1.0)
    
    if key == ord("2"):
        brightness = np.clip(brightness + 0.05, 0.0, 1.0)

    if key == ord("q"):
        break