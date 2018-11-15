import cv2

cap = cv2.VideoCapture('sora.mp4')
out = cv2.VideoWriter(
    'out.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    int(cap.get(cv2.CAP_PROP_FPS)),
    (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
)

if not cap.isOpened():
    print('fail open video')
    exit()
else:
    print('success open video')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)

cap.release()
