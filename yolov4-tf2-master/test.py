import cv2

cap = cv2.VideoCapture("D:/paCong/3.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("s", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()