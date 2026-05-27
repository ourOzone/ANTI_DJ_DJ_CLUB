import cv2
import numpy as np
import time
import os

BOARD_W = 10
BOARD_H = 7
SQUARE_SIZE = 2.0
CAPTURE_DURATION = 30
CAPTURE_INTERVAL = 0.5


def capture_single(cam_index):
    cap = cv2.VideoCapture(cam_index)
    frames = []
    start_time = time.time()
    last_capture = 0.0
    win_name = f"Camera {cam_index}"
    while True:
        elapsed = time.time() - start_time
        if elapsed >= CAPTURE_DURATION:
            break
        _, frame = cap.read()
        #frame = frame[:, ::-1, :]
        #cv2.putText(frame, f"{int(CAPTURE_DURATION - elapsed) + 1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow(win_name, frame)

        if elapsed - last_capture >= CAPTURE_INTERVAL:
            frames.append(frame.copy())
            last_capture = elapsed

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    return frames

def calibrate_camera(frames, cam_label):
    pattern_size = (BOARD_W, BOARD_H)
    objp = np.zeros((BOARD_H * BOARD_W, 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_W, 0:BOARD_H].T.reshape(-1, 2) * SQUARE_SIZE

    obj_points = []
    img_points = []

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    valid = 0
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if found:
            corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners_sub)
            valid += 1
    print(f"[{cam_label}] 유효 체커보드 감지: {valid}/{len(frames)} 프레임")

    if valid < 5:
        print(f"[{cam_label}] 유효 프레임이 너무 적습니다 (최소 5개 필요). 캘리브레이션을 건너뜁니다.")
        return None

    h, w = frames[0].shape[:2]
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, (w, h), None, None
    )

    return {
        "label": cam_label,
        "rms": rms,
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "valid_frames": valid,
        "total_frames": len(frames),
    }


def save_results(results, path="cali_result.txt"):
    import json
    data = {}
    for res in results:
        if res is None:
            continue
        key = res["label"].lower().replace(" ", "")
        data[key] = {
            "rms":    res["rms"],
            "matrix": res["camera_matrix"].tolist(),
            "dist":   res["dist_coeffs"].tolist(),
        }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"결과 저장 완료: {os.path.abspath(path)}")


frames0, frames1 = capture_single(0), capture_single(1)
print("캘리브레이션 중")
result0 = calibrate_camera(frames0, "Camera 0")
result1 = calibrate_camera(frames1, "Camera 1")
save_results([result0, result1])