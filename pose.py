import cv2
import numpy as np
import json


def load_calibration(path):
    with open(path, 'r') as f:
        calib = json.load(f)
    cams = {}
    for name, data in calib.items():
        cams[name] = {
            'matrix': np.array(data['matrix'], dtype=np.float64),
            'dist': np.array(data['dist'], dtype=np.float64),
        }
    return cams


def draw_box_on_chessboard(image, K, dist,
                           pattern_size=(10, 7),
                           square_size=1.0,
                           box_size=(3, 3, 3)):
    """반환: (out_image, found, rvec, tvec)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(
        gray, pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not found:
        return image, False, None, None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist)
    if not ok:
        return image, False, None, None

    w, d, h = box_size
    cx = (pattern_size[0] - 1) * square_size / 2 - w / 2
    cy = (pattern_size[1] - 1) * square_size / 2 - d / 2

    box_pts = np.float32([
        [cx,     cy,     0],
        [cx + w, cy,     0],
        [cx + w, cy + d, 0],
        [cx,     cy + d, 0],
        [cx,     cy,     -h],
        [cx + w, cy,     -h],
        [cx + w, cy + d, -h],
        [cx,     cy + d, -h],
    ])

    img_pts, _ = cv2.projectPoints(box_pts, rvec, tvec, K, dist)
    img_pts = img_pts.reshape(-1, 2).astype(np.int32)

    out = image.copy()
    cv2.drawContours(out, [img_pts[:4]], -1, (0, 255, 0), 2)
    for i in range(4):
        cv2.line(out, tuple(img_pts[i]), tuple(img_pts[i + 4]), (255, 0, 0), 2)
    cv2.drawContours(out, [img_pts[4:]], -1, (0, 0, 255), 2)
    cv2.drawFrameAxes(out, K, dist, rvec, tvec, square_size * 3, 2)

    return out, True, rvec, tvec


def hconcat_resize(img1, img2):
    """높이 다른 두 이미지도 안전하게 가로 연결."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = min(h1, h2)
    if h1 != h:
        img1 = cv2.resize(img1, (int(w1 * h / h1), h))
    if h2 != h:
        img2 = cv2.resize(img2, (int(w2 * h / h2), h))
    return np.hstack([img1, img2])


def draw_blinking_find(image, frame_count, period=10):
    """period 프레임마다 on/off 토글. 30fps면 약 0.33초 주기."""
    if (frame_count // period) % 2 == 0:
        cv2.putText(image, "FIND", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)


def save_pose(rvec0, tvec0, rvec1, tvec1, path='pose.txt'):
    data = {
        'camera0': {
            'rvec': rvec0.flatten().tolist(),
            'tvec': tvec0.flatten().tolist(),
        },
        'camera1': {
            'rvec': rvec1.flatten().tolist(),
            'tvec': tvec1.flatten().tolist(),
        },
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"[saved] {path}")


def main():
    cams = load_calibration('cali_result.txt')

    cap0 = cv2.VideoCapture(0)
    cap1 = cv2.VideoCapture(1)

    frame_count = 0

    try:
        while True:
            frame_count += 1

            ret0, f0 = cap0.read()
            ret1, f1 = cap1.read()
            if not (ret0 and ret1):
                continue

            out0, found0, rvec0, tvec0 = draw_box_on_chessboard(
                f0, cams['camera0']['matrix'], cams['camera0']['dist'])
            out1, found1, rvec1, tvec1 = draw_box_on_chessboard(
                f1, cams['camera1']['matrix'], cams['camera1']['dist'])

            # 각 카메라에서 검출되면 FIND 점멸
            if found0:
                draw_blinking_find(out0, frame_count)
            if found1:
                draw_blinking_find(out1, frame_count)

            combined = hconcat_resize(out0, out1)

            # 두 카메라 모두 검출 중일 때 안내 문구
            if found0 and found1:
                cv2.putText(combined, "BOTH FOUND - press [P] to save",
                            (30, combined.shape[0] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('stereo', combined)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC: 그냥 종료
                break

            if key == ord('p') and found0 and found1:
                save_pose(rvec0, tvec0, rvec1, tvec1)
                break

    finally:
        cap0.release()
        cap1.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()