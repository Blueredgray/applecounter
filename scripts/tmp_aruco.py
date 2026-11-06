"""Полный кадр + увеличенный участок с маркером, попытка декодирования."""
import cv2
from pathlib import Path

ROOT = Path(r"G:\opencode\soft\applecounter3")
VFC = ROOT / "data" / "raw" / "videofruit" / "Video Fruit Counting"
OUT = ROOT / "outputs" / "tmp_markers"

src = VFC / "Videos" / "Azure Kinect" / "210928_100624_k_r1_e_015_225_162.mp4"
cap = cv2.VideoCapture(str(src))
# boundary 5 = frame 2798 (маркер "16" был в центре)
cap.set(cv2.CAP_PROP_POS_FRAMES, 2798)
ok, frame = cap.read()
cap.release()
rot = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
h, w = rot.shape[:2]
print("rot size:", rot.shape)

# маркер был внизу-посредине; вырежем увеличенный участок
crop = rot[int(h*0.55):int(h*0.85), int(w*0.30):int(w*0.75)]
big = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
cv2.imwrite(str(OUT / "full_marker_zoom.png"), big)
print("saved zoom", big.shape)

# детекция на увеличенном изображении
gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
aruco = cv2.aruco
for dname in ("DICT_4X4_50", "DICT_5X5_100", "DICT_6X6_250", "DICT_APRILTAG_36h11",
              "DICT_APRILTAG_25h9", "DICT_APRILTAG_16h5"):
    d = aruco.getPredefinedDictionary(getattr(aruco, dname))
    det = aruco.ArucoDetector(d, aruco.DetectorParameters())
    corners, ids, rej = det.detectMarkers(gray)
    if ids is not None and len(ids):
        print("DECODED:", dname, ids.flatten().tolist())
