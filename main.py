import numpy as np
import cv2 as cv
import librosa
from pathlib import Path
import pygame
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import json
import time


BPS = 100


class CameraSystem:
    PLACEHOLDER_SIZE = (480, 640)

    def __init__(self, main_idx=2, left_idx=1, right_idx=0,
                 calib_path="cali_result.txt", pose_path="pose.txt"):
        self.main_cam = cv.VideoCapture(main_idx)
        self.left_cam = cv.VideoCapture(left_idx)
        self.right_cam = cv.VideoCapture(right_idx)

        cams_ok = all(c.isOpened() for c in [self.main_cam, self.left_cam, self.right_cam])
        if not cams_ok:
            print("[CameraSystem] 카메라 연결 안됨 — 음악 화면만 표시합니다.")

        try:
            self.calibration = self._load_calibration(calib_path)
            self.pose = self._load_pose(pose_path)
            files_ok = True
        except FileNotFoundError as e:
            print(f"[CameraSystem] 설정 파일 없음: {e}")
            self.calibration = None
            self.pose = None
            files_ok = False

        self.available = cams_ok and files_ok

    def _load_calibration(self, path):
        with open(path, 'r') as f:
            calib = json.load(f)
        return {
            name: {
                'matrix': np.array(data['matrix'], dtype=np.float64),
                'dist': np.array(data['dist'], dtype=np.float64),
            }
            for name, data in calib.items()
        }

    def _load_pose(self, path):
        with open(path, 'r') as f:
            pose = json.load(f)
        for cam in pose:
            pose[cam]['rvec'] = np.array(pose[cam]['rvec'], dtype=np.float64).reshape(3, 1)
            pose[cam]['tvec'] = np.array(pose[cam]['tvec'], dtype=np.float64).reshape(3, 1)
        return pose

    def _placeholder(self):
        h, w = self.PLACEHOLDER_SIZE
        img = np.full((h, w, 3), 40, dtype=np.uint8)
        cv.putText(img, "No Camera", (w // 2 - 90, h // 2),
                   cv.FONT_HERSHEY_SIMPLEX, 1.2, (80, 80, 80), 2)
        return img

    def get_cam_img(self):
        def read_or_placeholder(cam):
            if cam.isOpened():
                ok, img = cam.read()
                if ok:
                    return img
            return self._placeholder()

        return (
            read_or_placeholder(self.main_cam),
            read_or_placeholder(self.left_cam),
            read_or_placeholder(self.right_cam),
        )

    def release(self):
        self.main_cam.release()
        self.left_cam.release()
        self.right_cam.release()


class HandTracker:
    FINGER_IDS = {'thumb': 4, 'index': 8, 'middle': 12}

    def __init__(self, model_path="hand_landmarker.task"):
        try:
            options = vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                num_hands=1,
                min_hand_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            self.detector = vision.HandLandmarker.create_from_options(options)
            self.available = True
        except Exception as e:
            print(f"[HandTracker] 모델 로드 실패: {e}")
            self.detector = None
            self.available = False

    def detect(self, bgr_img):
        if not self.available:
            return None
        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
        )
        result = self.detector.detect(mp_img)
        if result.hand_landmarks:
            return result.hand_landmarks[0]
        return None

    def triangulate(self, landmarks0, landmarks1, shape0, shape1, calibration, pose):
        h0, w0 = shape0[:2]
        h1, w1 = shape1[:2]

        pts0, pts1 = [], []
        for idx in self.FINGER_IDS.values():
            l0, l1 = landmarks0[idx], landmarks1[idx]
            pts0.append([l0.x * w0, l0.y * h0])
            pts1.append([l1.x * w1, l1.y * h1])
        pts0 = np.array(pts0, dtype=np.float64)
        pts1 = np.array(pts1, dtype=np.float64)

        K0, d0 = calibration['camera0']['matrix'], calibration['camera0']['dist']
        K1, d1 = calibration['camera1']['matrix'], calibration['camera1']['dist']

        norm0 = cv.undistortPoints(pts0.reshape(-1, 1, 2), K0, d0).reshape(-1, 2)
        norm1 = cv.undistortPoints(pts1.reshape(-1, 1, 2), K1, d1).reshape(-1, 2)

        R0, _ = cv.Rodrigues(pose['camera0']['rvec'])
        R1, _ = cv.Rodrigues(pose['camera1']['rvec'])
        P0 = np.hstack([R0, pose['camera0']['tvec']])
        P1 = np.hstack([R1, pose['camera1']['tvec']])

        pts_4d = cv.triangulatePoints(P0, P1, norm0.T, norm1.T)
        pts_3d = (pts_4d[:3] / pts_4d[3]).T

        return {name: pts_3d[i] for i, name in enumerate(self.FINGER_IDS)}


class VirtualBox:
    def __init__(self, origin=(-3.0, -2.0), width=16, depth=10, height=1, hover_height=2):
        self.origin = origin
        self.width = width
        self.depth = depth
        self.height = height
        self.hover_height = hover_height

    def get_finger_state(self, finger_3d):
        cx, cy = self.origin
        x, y, z = finger_3d
        in_xy = (cx <= x <= cx + self.width) and (cy <= y <= cy + self.depth)
        if not in_xy:
            return None
        if -self.height <= z <= 0:
            return 'press'
        if -(self.height + self.hover_height) <= z <= -self.height:
            return 'hover'
        return None

    def draw(self, image, cam_name, calibration, pose, finger_state=None):
        COLOR_BOTTOM = (80, 80, 80)
        COLOR_TOP = {'press': (0, 0, 255), 'hover': (0, 165, 255)}.get(finger_state, (220, 220, 220))
        COLOR_FRONT = (60, 60, 200)
        COLOR_BACK = (200, 60, 60)
        COLOR_RIGHT = (60, 200, 60)
        COLOR_LEFT = (200, 200, 60)
        COLOR_EDGE = (0, 0, 0)
        EDGE_THICKNESS = 2
        ALPHA = 0.5

        K = calibration[cam_name]['matrix']
        dist = calibration[cam_name]['dist']
        rvec = pose[cam_name]['rvec']
        tvec = pose[cam_name]['tvec']
        cx, cy = self.origin
        w, d, h = self.width, self.depth, self.height

        box_pts = np.float32([
            [cx,     cy,      0], [cx + w, cy,      0], [cx + w, cy + d,  0], [cx,     cy + d,  0],
            [cx,     cy,     -h], [cx + w, cy,     -h], [cx + w, cy + d, -h], [cx,     cy + d, -h],
        ])
        img_pts, _ = cv.projectPoints(box_pts, rvec, tvec, K, dist)
        img_pts = img_pts.reshape(-1, 2).astype(np.int32)

        faces = [
            ([0, 1, 2, 3], COLOR_BOTTOM),
            ([4, 5, 6, 7], COLOR_TOP),
            ([0, 1, 5, 4], COLOR_FRONT),
            ([2, 3, 7, 6], COLOR_BACK),
            ([1, 2, 6, 5], COLOR_RIGHT),
            ([3, 0, 4, 7], COLOR_LEFT),
        ]

        R, _ = cv.Rodrigues(rvec)
        pts_cam = (R @ box_pts.T + tvec).T
        faces_sorted = sorted(
            [(np.mean(pts_cam[idx, 2]), idx, color) for idx, color in faces],
            key=lambda x: -x[0]
        )

        overlay = image.copy()
        for _, idx, color in faces_sorted:
            pts = img_pts[idx]
            cv.fillPoly(overlay, [pts], color)
            cv.polylines(overlay, [pts], True, COLOR_EDGE, EDGE_THICKNESS)

        return cv.addWeighted(overlay, ALPHA, image, 1 - ALPHA, 0)


class MusicVisualizer:
    # BGR: 첫박=파랑, 두박=보라, 세박=초록, 네박=남색
    BEAT_COLORS = [
        (255, 0, 0),
        (128, 0, 128),
        (0, 200, 0),
        (128, 0, 0),
    ]

    def __init__(self, bps=BPS):
        self.bps = bps

    def get_wave(self, music):
        file = Path(music.rsplit('.', 1)[0] + ".json")
        if not file.exists():
            return self._analyze(music)
        with open(file, 'r') as f:
            data = json.load(f)
        return data['bpm'], np.array(data['amplitudes'])

    def _analyze(self, music):
        y, sr = librosa.load(music, mono=True)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(np.atleast_1d(tempo)[0])

        sample = sr // self.bps
        bar_num = len(y) // sample
        y = y[:bar_num * sample].reshape(bar_num, sample)
        amplitudes = np.abs(y).max(axis=1)

        out_path = music.rsplit('.', 1)[0] + ".json"
        with open(out_path, 'w') as f:
            json.dump({'bpm': bpm, 'amplitudes': amplitudes.tolist()}, f)
        return bpm, amplitudes

    def get_wave_img(self, width, height, padding, music, time, length):
        w, h = width - 2 * padding, height - 2 * padding
        bpm, wave = self.get_wave(music)
        start = max(0, int(time * self.bps))
        end = min(len(wave), int((time + length) * self.bps))
        segment = wave[start:end]

        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        if len(segment) > 0:
            n = min(len(segment), w)
            amplitudes = segment[:n]
            bar_heights = (amplitudes * (h // 2)).astype(int)
            mid = padding + h // 2

            sub_beat_dur = 60.0 / bpm / 4
            col_times = time + np.arange(n) / self.bps
            sub_beat_idx = (col_times / sub_beat_dur).astype(int) % 4
            colors = np.array([self.BEAT_COLORS[i] for i in sub_beat_idx], dtype=np.uint8)

            rows = np.arange(padding, padding + h)[:, None]
            mask = np.abs(rows - mid) <= bar_heights[None, :]
            region = img[padding:padding + h, padding:n + padding]
            region[mask] = colors[np.where(mask)[1]]

        cv.rectangle(img, (padding, padding), (w + padding, padding + h), (0, 0, 0), 2)
        return img

    def get_music_cam(self, music1, time1, music2, time2, length=10):
        width = length * self.bps
        height = 200
        pad = 10
        img1 = self.get_wave_img(width, height, pad, music1, time1, length)
        img2 = self.get_wave_img(width, height, pad, music2, time2, length)
        return np.vstack([img1, img2])


class MusicPlayer:
    def __init__(self, sample_rate=44100):
        pygame.mixer.pre_init(sample_rate, -16, 2, 2048)
        pygame.mixer.init()
        pygame.mixer.set_num_channels(2)
        self.sample_rate = sample_rate
        self._audio = [None, None]
        self.channels = [pygame.mixer.Channel(0), pygame.mixer.Channel(1)]
        self._start_time = [None, None]
        self._offset = [0.0, 0.0]
        self.is_playing = [False, False]

    def load(self, track_idx, path):
        print(f"[MusicPlayer] 로딩 중: {path}")
        y, sr = librosa.load(path, sr=self.sample_rate, mono=False)
        if y.ndim == 1:
            y = np.stack([y, y])
        self._audio[track_idx] = (y.T * 32767).astype(np.int16)
        self._offset[track_idx] = 0.0
        self.is_playing[track_idx] = False
        print(f"[MusicPlayer] 로딩 완료: {path}")

    def play(self, track_idx):
        audio = self._audio[track_idx]
        if audio is None:
            return
        offset_samples = int(self._offset[track_idx] * self.sample_rate)
        sliced = audio[offset_samples:]
        if len(sliced) == 0:
            return
        self.channels[track_idx].play(pygame.sndarray.make_sound(sliced))
        self._start_time[track_idx] = time.time()
        self.is_playing[track_idx] = True

    def pause(self, track_idx):
        if not self.is_playing[track_idx]:
            return
        self._offset[track_idx] += time.time() - self._start_time[track_idx]
        self.channels[track_idx].stop()
        self.is_playing[track_idx] = False

    def toggle(self, track_idx):
        if self.is_playing[track_idx]:
            self.pause(track_idx)
        else:
            self.play(track_idx)

    def get_pos(self, track_idx):
        if self.is_playing[track_idx]:
            if not self.channels[track_idx].get_busy():
                self.is_playing[track_idx] = False
            else:
                return self._offset[track_idx] + (time.time() - self._start_time[track_idx])
        return self._offset[track_idx]

    def stop(self, track_idx):
        self.channels[track_idx].stop()
        self._offset[track_idx] = 0.0
        self.is_playing[track_idx] = False


def four_corner_layout(top_left, top_right, bottom_left, bottom_right, bg=(30, 30, 30)):
    tl_h, tl_w = top_left.shape[:2]
    tr_h, tr_w = top_right.shape[:2]
    bl_h, bl_w = bottom_left.shape[:2]
    br_h, br_w = bottom_right.shape[:2]

    canvas_w = max(tl_w + tr_w, bl_w + br_w)
    canvas_h = max(tl_h + bl_h, tr_h + br_h)

    canvas = np.full((canvas_h, canvas_w, 3), bg, dtype=np.uint8)
    canvas[0:tl_h, 0:tl_w] = top_left
    canvas[0:tr_h, canvas_w - tr_w:canvas_w] = top_right
    canvas[canvas_h - bl_h:canvas_h, 0:bl_w] = bottom_left
    canvas[canvas_h - br_h:canvas_h, canvas_w - br_w:canvas_w] = bottom_right
    return canvas


LEFT_TRACK  = "sound/left/cant_stop.mp3"
RIGHT_TRACK = "sound/right/gangnam_style.mp3"


def main():
    cameras = CameraSystem()
    tracker = HandTracker()
    box = VirtualBox()
    visualizer = MusicVisualizer()

    player = MusicPlayer()
    player.load(0, LEFT_TRACK)
    player.load(1, RIGHT_TRACK)
    player.play(0)
    player.play(1)

    while True:
        music_img = visualizer.get_music_cam(
            LEFT_TRACK,  player.get_pos(0),
            RIGHT_TRACK, player.get_pos(1),
        )

        if cameras.available:
            main_img, left_img, right_img = cameras.get_cam_img()

            finger_state = None
            lm0 = tracker.detect(right_img)
            lm1 = tracker.detect(left_img)
            if lm0 and lm1:
                fingers_3d = tracker.triangulate(
                    lm0, lm1,
                    right_img.shape, left_img.shape,
                    cameras.calibration, cameras.pose
                )
                finger_state = box.get_finger_state(fingers_3d['index'])

            right_img = box.draw(right_img, 'camera0', cameras.calibration, cameras.pose, finger_state)
            left_img = box.draw(left_img, 'camera1', cameras.calibration, cameras.pose, finger_state)

            canvas = four_corner_layout(
                top_left=left_img,
                top_right=right_img,
                bottom_left=main_img,
                bottom_right=music_img,
            )
        else:
            canvas = music_img

        cv.imshow('all', canvas)
        key = cv.waitKey(1) & 0xFF
        if key == 27:           # ESC — 종료
            break
        elif key == ord('1'):   # 1 — 왼쪽 트랙 재생/일시정지
            player.toggle(0)
        elif key == ord('2'):   # 2 — 오른쪽 트랙 재생/일시정지
            player.toggle(1)

    player.stop(0)
    player.stop(1)
    cameras.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
