import cv2

cap = cv2.VideoCapture(0)

while cap.isOpened() :
    ret, frame = cap.read()
    flipped_frame = cv2.flip(frame, 1)
    if not ret :
        break

    cv2.imshow('frame', flipped_frame)
    if cv2.waitKey(1) & 0xFF == ord('q') :
        break

cap.release()
cv2.destroyAllWindows()
