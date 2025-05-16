import cv2
cap = cv2.VideoCapture('sample.mp4') 
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 480))
    fgmask = fgbg.apply(frame)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue 
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow('Motion Tracking', frame)
    cv2.imshow('Foreground Mask', fgmask)
    if cv2.waitKey(30) & 0xFF == 27: 
        break
cap.release()
cv2.destroyAllWindows()