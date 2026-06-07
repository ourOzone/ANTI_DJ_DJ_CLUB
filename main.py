import math
import threading
import queue
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import cv2 as cv
import librosa
import pygame
import sounddevice as sd
import json
import time
from pathlib import Path

BPS        = 100
WINDOW_SEC = 10.0
CAM_H = 300   # 두 배 크기
CAM_W = 400   # 두 배 크기
CAM2_W = int(CAM_W * 0.4)  # 카메라 3
CAM2_H = int(CAM_H * 0.4)
WAVEFORM_W = CAM_W * 2
BEAT_COLORS = np.array([
    (  0, 165, 255),   # 첫박  — 주황
    (  0, 120, 220),   # 둘째박 — 진한 주황
    ( 80, 200, 255),   # 셋째박 — 연한 주황
    (  0,  80, 200),   # 넷째박 — 번트 오렌지
], dtype=np.uint8)


def find_mp3(directory):
    files = sorted(Path(directory).glob("*.mp3"))
    if not files:
        raise FileNotFoundError(f"MP3 파일 없음: {directory}")
    return files[0]


def analyze(path):
    cache = path.with_suffix(".json")
    if cache.exists():
        data = json.loads(cache.read_text())
        return data["bpm"], np.array(data["amplitudes"]), data["beat_offset"]

    y, sr = librosa.load(str(path), mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    bpm = int(round(float(np.atleast_1d(tempo)[0])))

    beat_dur    = 60.0 / bpm
    beat_offset = float(librosa.frames_to_time(beat_frames, sr=sr)[0]) % beat_dur

    sample = sr // BPS
    count  = len(y) // sample
    amplitudes = np.abs(y[: count * sample].reshape(count, sample)).max(axis=1)

    cache.write_text(json.dumps({
        "bpm":         bpm,
        "amplitudes":  amplitudes.tolist(),
        "beat_offset": beat_offset,
    }))
    return bpm, amplitudes, beat_offset


def draw_waveform(amplitudes, bpm, beat_offset, label,
                  pos=0.0, loop=None, cue=None, width=WAVEFORM_W, height=150, pad=10):
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    w, h = width - 2 * pad, height - 2 * pad
    mid  = pad + h // 2

    start   = int(pos * BPS)
    end     = start + int(WINDOW_SEC * BPS)
    segment = amplitudes[start: min(end, len(amplitudes))]
    n       = min(len(segment), w)

    if n > 0:
        bar_heights = (segment[:n] * (h // 2)).astype(int)
        beat_dur    = 60.0 / bpm
        col_times   = (start + np.arange(n)) / BPS
        beat_phase  = (col_times - beat_offset) / beat_dur
        colors      = BEAT_COLORS[np.floor(beat_phase).astype(int) % 4]

        rows = np.arange(height)[:, None]                        # (H, 1)
        mask = np.abs(rows - mid) <= bar_heights[None, :n]       # (H, n) 브로드캐스팅
        img[:, pad:pad + n][mask] = colors[np.where(mask)[1]]

    # 루프 범위 오버레이
    if loop is not None:
        loop_start, loop_end = loop
        lx1 = pad + int((loop_start - pos) * BPS)
        lx2 = pad + int((loop_end   - pos) * BPS)
        lx1 = max(pad, min(pad + w, lx1))
        lx2 = max(pad, min(pad + w, lx2))
        if lx1 < lx2:
            overlay = img.copy()
            cv.rectangle(overlay, (lx1, pad), (lx2, pad + h), (0, 210, 210), -1)
            cv.addWeighted(overlay, 0.3, img, 0.7, 0, img)
            cv.line(img, (lx1, pad), (lx1, pad + h), (0, 160, 160), 2)
            cv.line(img, (lx2, pad), (lx2, pad + h), (0, 160, 160), 2)

    # 큐포인트 마디 하이라이트
    if cue is not None:
        bar_dur_sec = 60.0 / bpm  # 1박
        qx1 = pad + int((cue              - pos) * BPS)
        qx2 = pad + int((cue + bar_dur_sec - pos) * BPS)
        qx1 = max(pad, min(pad + w, qx1))
        qx2 = max(pad, min(pad + w, qx2))
        if qx1 < qx2:
            overlay = img.copy()
            cv.rectangle(overlay, (qx1, pad), (qx2, pad + h), (0, 220, 255), -1)
            cv.addWeighted(overlay, 0.28, img, 0.72, 0, img)
        cue_px = pad + int((cue - pos) * BPS)
        if pad <= cue_px <= pad + w:
            cv.line(img, (cue_px, pad), (cue_px, pad + h), (0, 255, 255), 2)

    cv.line(img, (pad, pad), (pad, pad + h), (0, 0, 0), 2)
    cv.rectangle(img, (pad, pad), (pad + w, pad + h), (0, 0, 0), 1)
    cv.putText(img, label, (pad + 8, pad + 18),
               cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return img


class ThreadedCamera:
    def __init__(self, idx):
        self._cap   = cv.VideoCapture(idx)
        self._frame = None
        self._lock  = threading.Lock()
        self._alive = self._cap.isOpened()
        if self._alive:
            threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self._alive:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def release(self):
        self._alive = False
        self._cap.release()


class ScratchEngine:
    """sounddevice 콜백으로 가변 속도 연속 재생 — 진짜 스크래치 사운드."""

    BLOCK = 128

    def __init__(self, sample_rate=44100):
        self._sr       = sample_rate
        self._audio    = None   # (N, 2) float32, [-1, 1]
        self._pos      = 0.0    # 현재 위치 (샘플 단위)
        self._rate     = 0.0    # 목표 rate
        self._rate_cur = 0.0    # 현재 smoothed rate
        self._lock     = threading.Lock()
        # 15ms 시상수 — 블록 경계 클릭 제거
        self._alpha = math.exp(-1.0 / (0.015 * sample_rate))
        self._stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=2,
            dtype='float32',
            blocksize=self.BLOCK,
            callback=self._cb,
        )
        self._stream.start()

    def _cb(self, outdata, frames, time_info, status):
        with self._lock:
            audio    = self._audio
            pos      = self._pos
            rate_tgt = self._rate
            rate_cur = self._rate_cur

        if audio is None or (abs(rate_tgt) < 0.001 and abs(rate_cur) < 0.001):
            outdata[:] = 0
            return

        # 샘플 단위 지수 보간으로 rate를 부드럽게 전환 → 클릭 없음
        decay = self._alpha ** np.arange(frames, dtype=np.float64)
        rates = rate_tgt + (rate_cur - rate_tgt) * decay

        offsets = np.concatenate([[0.0], np.cumsum(rates[:-1])])
        src = np.clip(pos + offsets, 0, len(audio) - 2)
        idx  = src.astype(np.int64)
        frac = (src - idx)[:, None]
        outdata[:] = (audio[idx] * (1 - frac) + audio[idx + 1] * frac).astype(np.float32)

        with self._lock:
            self._pos     = float(np.clip(pos + rates.sum(), 0, len(audio) - 1))
            self._rate_cur = float(rate_tgt + (rate_cur - rate_tgt) * self._alpha ** frames)

    def load(self, audio_int16):
        with self._lock:
            self._audio    = audio_int16.astype(np.float32) / 32768.0
            self._rate_cur = 0.0

    def set_pos(self, pos_sec):
        with self._lock:
            self._pos = pos_sec * self._sr

    def get_pos(self):
        with self._lock:
            return self._pos / self._sr

    def set_rate(self, rate):
        with self._lock:
            self._rate = rate

    def close(self):
        self._stream.stop()
        self._stream.close()


class MusicPlayer:
    SAMPLE_RATE = 44100

    def __init__(self):
        pygame.mixer.pre_init(self.SAMPLE_RATE, -16, 2, 2048)
        pygame.mixer.init()
        pygame.mixer.set_num_channels(2)
        self._audio      = [None, None]
        self._channels   = [pygame.mixer.Channel(0), pygame.mixer.Channel(1)]
        self._start_time = [None, None]
        self._offset     = [0.0, 0.0]
        self.is_playing  = [False, False]
        self._loop         = [False, False]
        self._loop_start   = [0.0, 0.0]
        self._loop_end     = [0.0, 0.0]
        self._loop_sound   = [None, None]  # pygame Sound 레퍼런스 유지 (GC 방지)
        self._loop_pending        = [False, False]  # loop_end 도달 전까지 대기
        self._scratch             = [False, False]
        self._scratch_was_playing = [False, False]
        self._scratch_engine = ScratchEngine(self.SAMPLE_RATE)
        self._cue = [None, None]

    def load(self, track_idx, path):
        print(f"  오디오 로드: {path.name}")
        y, sr = librosa.load(str(path), sr=self.SAMPLE_RATE, mono=False)
        if y.ndim == 1:
            y = np.stack([y, y])
        self._audio[track_idx]     = (y.T * 32767).astype(np.int16)
        self._offset[track_idx]    = 0.0
        self.is_playing[track_idx] = False

    # ── 루프 ────────────────────────────────────────────────

    def toggle_loop(self, track_idx, bar_dur, beat_offset):
        if self._loop[track_idx]:
            self._disable_loop(track_idx)
        else:
            self._enable_loop(track_idx, bar_dur, beat_offset)

    def _enable_loop(self, track_idx, bar_dur, beat_offset):
        pos        = self.get_pos(track_idx)
        # 현재 위치가 속한 마디의 시작으로 snap
        bars       = math.floor((pos - beat_offset) / bar_dur)
        loop_start = beat_offset + bars * bar_dur
        loop_end   = min(loop_start + bar_dur,
                         len(self._audio[track_idx]) / self.SAMPLE_RATE)
        self._loop_start[track_idx] = loop_start
        self._loop_end[track_idx]   = loop_end
        self._loop[track_idx] = True
        if self.is_playing[track_idx]:
            self._loop_pending[track_idx] = True  # loop_end까지 현재 재생 유지
        # 정지 중이면 toggle()로 재생 시 _play_loop() 호출됨


    def update(self):
        for i in range(2):
            if self._loop_pending[i] and self.is_playing[i]:
                if self.get_pos(i) >= self._loop_end[i]:
                    self._loop_pending[i] = False
                    self._play_loop(i)

    def _disable_loop(self, track_idx):
        was_playing = self.is_playing[track_idx]
        if was_playing and not self._loop_pending[track_idx]:
            # 루프 내 현재 위치 계산 후 일반 재생으로 전환
            elapsed  = time.time() - self._start_time[track_idx]
            loop_len = self._loop_end[track_idx] - self._loop_start[track_idx]
            self._offset[track_idx] = (
                self._loop_start[track_idx] + (elapsed % loop_len)
            )
            self._channels[track_idx].stop()
            self.is_playing[track_idx] = False
        self._loop[track_idx]         = False
        self._loop_pending[track_idx] = False
        self._loop_sound[track_idx]   = None
        if was_playing and not self.is_playing[track_idx]:
            self._play_normal(track_idx)

    def _play_loop(self, track_idx):
        start_s = int(self._loop_start[track_idx] * self.SAMPLE_RATE)
        end_s   = int(self._loop_end[track_idx]   * self.SAMPLE_RATE)
        seg     = np.ascontiguousarray(self._audio[track_idx][start_s:end_s])
        sound   = pygame.sndarray.make_sound(seg)
        self._loop_sound[track_idx] = sound
        self._channels[track_idx].stop()
        self._channels[track_idx].play(sound, loops=-1)
        self._start_time[track_idx] = time.time()
        self.is_playing[track_idx]  = True

    # ── 재생/정지 ────────────────────────────────────────────

    def toggle(self, track_idx):
        if self.is_playing[track_idx]:
            self._stop(track_idx)
        else:
            if self._loop[track_idx]:
                self._play_loop(track_idx)
            else:
                self._play_normal(track_idx)

    def _play_normal(self, track_idx):
        audio = self._audio[track_idx]
        if audio is None:
            return
        offset_samples = int(self._offset[track_idx] * self.SAMPLE_RATE)
        sliced = audio[offset_samples:]
        if len(sliced) == 0:
            return
        self._channels[track_idx].play(
            pygame.sndarray.make_sound(np.ascontiguousarray(sliced))
        )
        self._start_time[track_idx] = time.time()
        self.is_playing[track_idx]  = True

    def _stop(self, track_idx):
        if not self.is_playing[track_idx]:
            return
        elapsed = time.time() - self._start_time[track_idx]
        if self._loop[track_idx] and not self._loop_pending[track_idx]:
            loop_len = self._loop_end[track_idx] - self._loop_start[track_idx]
            self._offset[track_idx] = (
                self._loop_start[track_idx] + (elapsed % loop_len)
            )
        else:
            self._offset[track_idx] += elapsed
        self._channels[track_idx].stop()
        self.is_playing[track_idx]    = False
        self._loop_pending[track_idx] = False

    def get_pos(self, track_idx):
        if self._scratch[track_idx]:
            return self._scratch_engine.get_pos()
        if self.is_playing[track_idx]:
            if not self._channels[track_idx].get_busy():
                self.is_playing[track_idx] = False
            else:
                elapsed = time.time() - self._start_time[track_idx]
                if self._loop[track_idx] and not self._loop_pending[track_idx]:
                    loop_len = self._loop_end[track_idx] - self._loop_start[track_idx]
                    return self._loop_start[track_idx] + (elapsed % loop_len)
                return self._offset[track_idx] + elapsed
        return self._offset[track_idx]

    # ── 스크래치 ─────────────────────────────────────────────

    def start_scratch(self, track_idx):
        self._scratch_was_playing[track_idx] = self.is_playing[track_idx]
        if self.is_playing[track_idx]:
            self._stop(track_idx)
        self._scratch[track_idx] = True
        self._scratch_engine.load(self._audio[track_idx])
        self._scratch_engine.set_pos(self._offset[track_idx])
        self._scratch_engine.set_rate(0.0)

    def scratch(self, track_idx, delta_sec, delta_time):
        if not self._scratch[track_idx]:
            return
        rate = delta_sec / max(delta_time, 0.001)
        rate = max(-8.0, min(8.0, rate))
        self._scratch_engine.set_rate(rate)

    def end_scratch(self, track_idx):
        self._scratch_engine.set_rate(0.0)
        self._scratch[track_idx] = False
        # 큐포인트가 있으면 항상 그 위치로 복귀
        self._offset[track_idx] = (self._cue[track_idx]
                                   if self._cue[track_idx] is not None
                                   else self._scratch_engine.get_pos())
        if self._scratch_was_playing[track_idx]:
            self._play_normal(track_idx)

    def set_cue(self, track_idx, bar_dur, beat_offset):
        """현재 위치가 속한 박의 시작으로 큐포인트를 설정."""
        pos      = self.get_pos(track_idx)
        beat_dur = bar_dur / 4
        beats    = math.floor((pos - beat_offset) / beat_dur)
        self._cue[track_idx] = beat_offset + beats * beat_dur

    def get_cue(self, track_idx):
        return self._cue[track_idx]

    def get_loop(self, track_idx):
        """루프 중이면 (loop_start, loop_end), 아니면 None."""
        if self._loop[track_idx]:
            return self._loop_start[track_idx], self._loop_end[track_idx]
        return None

    def stop_all(self):
        for i in range(2):
            self._stop(i)
        self._scratch_engine.close()


class HandTracker:
    INDEX_ID = 8  # MediaPipe 검지 끝 랜드마크

    def __init__(self, model_path="hand_landmarker.task"):
        self.available = False
        self.detector = None
        for use_gpu in (True, False):
            try:
                base_opts = (
                    BaseOptions(model_asset_path=model_path,
                                delegate=BaseOptions.Delegate.GPU)
                    if use_gpu else BaseOptions(model_asset_path=model_path)
                )
                opts = vision.HandLandmarkerOptions(
                    base_options=base_opts,
                    num_hands=2,
                    min_hand_detection_confidence=0.4,
                    min_hand_presence_confidence=0.4,
                    min_tracking_confidence=0.3,
                )
                self.detector = vision.HandLandmarker.create_from_options(opts)
                self.available = True
                print(f"[HandTracker] {'GPU' if use_gpu else 'CPU'} 모드")
                break
            except Exception as e:
                if not use_gpu:
                    print(f"[HandTracker] 초기화 실패: {e}")

    def detect(self, bgr_img):
        """검출된 손들을 {'left': landmarks, 'right': landmarks} 형태로 반환."""
        if not self.available:
            return {}
        rgba = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGBA)
        result = self.detector.detect(
            mp.Image(image_format=mp.ImageFormat.SRGBA, data=rgba)
        )
        return {
            hn[0].category_name.lower(): lm
            for lm, hn in zip(result.hand_landmarks, result.handedness)
        }

    def _triangulate(self, lm0, lm1, shape0, shape1, cams, pose):
        h0, w0 = shape0[:2]
        h1, w1 = shape1[:2]
        pt0 = np.array([[lm0[self.INDEX_ID].x * w0, lm0[self.INDEX_ID].y * h0]], dtype=np.float64)
        pt1 = np.array([[lm1[self.INDEX_ID].x * w1, lm1[self.INDEX_ID].y * h1]], dtype=np.float64)
        K0, d0 = cams['camera0']['matrix'], cams['camera0']['dist']
        K1, d1 = cams['camera1']['matrix'], cams['camera1']['dist']
        n0 = cv.undistortPoints(pt0.reshape(-1, 1, 2), K0, d0).reshape(-1, 2)
        n1 = cv.undistortPoints(pt1.reshape(-1, 1, 2), K1, d1).reshape(-1, 2)
        R0, _ = cv.Rodrigues(pose['camera0']['rvec'])
        R1, _ = cv.Rodrigues(pose['camera1']['rvec'])
        P0 = np.hstack([R0, pose['camera0']['tvec']])
        P1 = np.hstack([R1, pose['camera1']['tvec']])
        pts_4d = cv.triangulatePoints(P0, P1, n0.T, n1.T)
        return (pts_4d[:3] / pts_4d[3]).T[0]

    def get_finger_pts(self, hands0, hands1, shape0, shape1, cams, pose):
        """두 카메라 결과에서 각 손의 검지 3D 위치 목록을 반환."""
        pts = []
        matched = set()
        # 1차: 핸드니스(left/right) 기반 매칭
        for side in ('left', 'right'):
            if side in hands0 and side in hands1:
                try:
                    pts.append(self._triangulate(
                        hands0[side], hands1[side], shape0, shape1, cams, pose
                    ))
                    matched.add(side)
                except Exception:
                    pass

        # 2차 폴백: 핸드니스 불일치 손이 남아있을 때 단독 삼각측량 시도
        unmatched0 = [lm for s, lm in hands0.items() if s not in matched]
        unmatched1 = [lm for s, lm in hands1.items() if s not in matched]
        for lm0, lm1 in zip(unmatched0, unmatched1):
            try:
                pts.append(self._triangulate(lm0, lm1, shape0, shape1, cams, pose))
            except Exception:
                pass
        return pts


class AsyncHandTracker:
    def __init__(self, tracker):
        self._tracker = tracker
        self._result = ({}, {})
        self._lock = threading.Lock()
        self._q = queue.Queue(maxsize=1)
        if tracker.available:
            threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            img0, img1 = self._q.get()
            r0 = self._tracker.detect(img0)
            r1 = self._tracker.detect(img1)
            with self._lock:
                self._result = (r0, r1)

    def submit(self, img0, img1):
        try:
            self._q.put_nowait((img0.copy(), img1.copy()))
        except queue.Full:
            pass

    def get(self):
        with self._lock:
            return self._result


def get_section(pt, box_size=(15, 10, 1.125), pattern_size=(10, 7), square_size=1.0):
    """3D 점이 속한 구획을 (track_idx, zone) 으로 반환. 박스 밖이면 None.
    zone: 'top' | 'bot1'(재생/정지) | 'bot2'(루프) | 'bot3'(킬스위치)
    """
    w, d, _ = box_size
    cx = (pattern_size[0] - 1) * square_size / 2 - w / 2
    cy = (pattern_size[1] - 1) * square_size / 2 - d / 2
    x_mid   = cx + w / 2
    y_split = cy + w / 2

    x, y = float(pt[0]), float(pt[1])
    if not (cx <= x <= cx + w and cy <= y <= cy + d):
        return None
    track = 0 if x < x_mid else 1

    if y < y_split:
        return track, 'top'

    # 하단 스트립을 x축 기준 3등분
    x_origin = cx if track == 0 else x_mid
    sub_w    = (w / 2) / 3
    rel_x    = x - x_origin
    if rel_x < sub_w:
        zone = 'bot1'
    elif rel_x < sub_w * 2:
        zone = 'bot2'
    else:
        zone = 'bot3'
    return track, zone


def draw_top_view(finger_pts, active_states=None, angles=None, box_size=(15, 10, 1.125),
                  pattern_size=(10, 7), square_size=1.0,
                  canvas_w=500, canvas_h=400, margin=40):
    """DJ 탑뷰: 바이닐 디스크 + 버튼 패널."""
    if active_states is None:
        active_states = [[False, False, False], [False, False, False]]
    if angles is None:
        angles = [0.0, 0.0]

    w, d, _ = box_size
    cx = (pattern_size[0] - 1) * square_size / 2 - w / 2
    cy = (pattern_size[1] - 1) * square_size / 2 - d / 2
    x_mid   = cx + w / 2
    y_split = cy + w / 2
    sub_w   = (w / 2) / 3

    x_min, x_max = cx - 3, cx + w + 3
    y_min, y_max = cy - 3, cy + d + 3
    scale = min(
        (canvas_w - 2 * margin) / (x_max - x_min),
        (canvas_h - 2 * margin) / (y_max - y_min),
    )

    def to_px(x, y):
        return (int(margin + (x - x_min) * scale),
                int(margin + (y - y_min) * scale))

    # ── 배경 ────────────────────────────────────────────────────
    img = np.full((canvas_h, canvas_w, 3), (248, 248, 250), dtype=np.uint8)

    # 덱 패널 배경
    for x0, x1 in [(cx, x_mid), (x_mid, cx + w)]:
        p0, p1 = to_px(x0, cy), to_px(x1, cy + d)
        cv.rectangle(img, p0, p1, (22, 22, 22), -1)

    # ── 바이닐 디스크 ────────────────────────────────────────────
    r_world    = (w / 2) / 2 * 0.8
    r_px       = int(r_world * scale)
    label_r    = max(8, r_px // 3)
    spindle_r  = max(2, r_px // 16)
    n_grooves  = 16
    deck_label_colors = [(0, 140, 255), (0, 140, 255)]  # BGR: 주황

    for side, x0 in enumerate((cx, x_mid)):
        cp = to_px(x0 + w / 4, cy + w / 4)

        # 외곽 림
        cv.circle(img, cp, r_px,     (55, 55, 55), 2)
        # 디스크 몸체
        cv.circle(img, cp, r_px - 2, (12, 12, 12), -1)
        # 그루브 링 (교대로 밝기 다름)
        for k in range(n_grooves, 0, -1):
            gr = int((r_px - 4) * k / n_grooves)
            c  = 36 if k % 2 == 0 else 20
            cv.circle(img, cp, gr, (c, c, c), 1)
        # 중앙 레이블 스티커
        cv.circle(img, cp, label_r,     deck_label_colors[side], -1)
        cv.circle(img, cp, label_r,     (70, 70, 70), 1)
        # 스핀들 구멍
        cv.circle(img, cp, spindle_r, (180, 180, 180), -1)
        # 하이라이트 반사 (회전 적용)
        cv.ellipse(img, cp, (r_px - 4, r_px - 4), angles[side], 210, 260,
                   (60, 60, 60), 2)
        # 레이블 회전 마커
        rad = math.radians(angles[side])
        mx = int(cp[0] + label_r * 0.65 * math.cos(rad))
        my = int(cp[1] + label_r * 0.65 * math.sin(rad))
        cv.line(img, cp, (mx, my), (210, 210, 210), 1)

    # ── 버튼 패널 ────────────────────────────────────────────────
    GAP    = 4
    BORDER = 2

    def draw_button(x0, y0, x1, y1, lbl, active):
        p0 = (to_px(x0, y0)[0] + GAP, to_px(x0, y0)[1] + GAP)
        p1 = (to_px(x1, y1)[0] - GAP, to_px(x1, y1)[1] - GAP)
        border = (0, 140, 255) if active else (80, 80, 80)
        cv.rectangle(img, p0, p1, (22, 22, 22), -1)
        cv.rectangle(img, p0, p1, border, BORDER)
        cv.line(img, (p0[0] + 2, p0[1] + 2), (p1[0] - 2, p0[1] + 2),
                (55, 55, 55), 1)

    bot_labels = ["PLAY", "LOOP", "CUE"]
    for side, x0 in enumerate([cx, x_mid]):
        for i, lbl in enumerate(bot_labels):
            draw_button(x0 + sub_w * i, y_split,
                        x0 + sub_w * (i + 1), cy + d,
                        lbl, active=active_states[side][i])

    # ── 외곽선 & 구분선 ─────────────────────────────────────────
    corners = np.array([to_px(cx, cy), to_px(cx + w, cy),
                        to_px(cx + w, cy + d), to_px(cx, cy + d)], dtype=np.int32)
    cv.polylines(img, [corners], True, (60, 60, 60), 2)
    cv.line(img, to_px(x_mid, cy), to_px(x_mid, cy + d), (50, 50, 50), 1)

    # ── 손가락 점 ────────────────────────────────────────────────
    for pt in finger_pts:
        px, py = to_px(float(pt[0]), float(pt[1]))
        if 0 <= px < canvas_w and 0 <= py < canvas_h:
            if float(pt[2]) > -1.0:  # 클릭
                cv.circle(img, (px, py), 10, (0, 220, 255), -1)
                cv.circle(img, (px, py), 10, (255, 255, 255),  1)
            else:
                cv.circle(img, (px, py),  8, (50, 100, 200), -1)
                cv.circle(img, (px, py),  8, (130, 160, 255), 1)

    return img


def compute_ar_H(K, dist, rvec, tvec, orig_w, orig_h, dst_w, dst_h,
                 box_size=(15, 10, 1.125), pattern_size=(10, 7), square_size=1.0,
                 canvas_w=500, canvas_h=400, margin=40):
    """AR 호모그래피를 한 번만 계산해 반환. pose 고정이므로 루프 밖에서 호출."""
    w, d, h = box_size
    cx = (pattern_size[0] - 1) * square_size / 2 - w / 2
    cy = (pattern_size[1] - 1) * square_size / 2 - d / 2

    top_3d = np.float32([
        [cx,     cy,     -h],
        [cx + w, cy,     -h],
        [cx + w, cy + d, -h],
        [cx,     cy + d, -h],
    ])
    img_pts, _ = cv.projectPoints(top_3d, rvec, tvec, K, dist)
    dst_pts = img_pts.reshape(-1, 2).astype(np.float32)
    dst_pts[:, 0] *= dst_w / orig_w
    dst_pts[:, 1] *= dst_h / orig_h

    x_min = cx - 3
    y_min = cy - 3
    x_max = cx + w + 3
    y_max = cy + d + 3
    scale = min(
        (canvas_w - 2 * margin) / (x_max - x_min),
        (canvas_h - 2 * margin) / (y_max - y_min),
    )
    src_pts = np.float32([
        [margin + (cx     - x_min) * scale, margin + (cy     - y_min) * scale],
        [margin + (cx + w - x_min) * scale, margin + (cy     - y_min) * scale],
        [margin + (cx + w - x_min) * scale, margin + (cy + d - y_min) * scale],
        [margin + (cx     - x_min) * scale, margin + (cy + d - y_min) * scale],
    ])

    try:
        H, _ = cv.findHomography(src_pts, dst_pts)
        return H
    except cv.error:
        return None


def apply_ar_overlay(frame, topview_img, H, alpha=0.7):
    """사전 계산된 H로 AR 합성. warpPerspective 1회 (BGRA)."""
    if H is None:
        return frame
    fh, fw = frame.shape[:2]

    bg_mask = cv.inRange(topview_img,
                         np.array([240, 240, 240], dtype=np.uint8),
                         np.array([255, 255, 255], dtype=np.uint8))
    bgra = cv.cvtColor(topview_img, cv.COLOR_BGR2BGRA)
    bgra[:, :, 3] = cv.bitwise_not(bg_mask)

    warped = cv.warpPerspective(bgra, H, (fw, fh))
    mask = warped[:, :, 3] > 128
    warped_bgr = warped[:, :, :3]

    blended = cv.addWeighted(warped_bgr, alpha, frame, 1 - alpha, 0)
    result = frame.copy()
    result[mask] = blended[mask]
    return result


def load_pose_and_calib(calib_path='cali_result.txt', pose_path='pose.txt'):
    """cali_result.txt와 pose.txt를 읽어 (cams, pose) 반환. 파일 없으면 (None, None)."""
    try:
        with open(calib_path) as f:
            calib = json.load(f)
        cams = {
            name: {
                'matrix': np.array(data['matrix'], dtype=np.float64),
                'dist':   np.array(data['dist'],   dtype=np.float64),
            }
            for name, data in calib.items()
        }
        with open(pose_path) as f:
            pose_data = json.load(f)
        pose = {
            name: {
                'rvec': np.array(data['rvec'], dtype=np.float64).reshape(3, 1),
                'tvec': np.array(data['tvec'], dtype=np.float64).reshape(3, 1),
            }
            for name, data in pose_data.items()
        }
        return cams, pose
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return None, None


def draw_booth_box(frame, K, dist, rvec, tvec,
                   pattern_size=(10, 7), square_size=1.0,
                   box_size=(15, 10, 1.125)):
    """pose.txt의 rvec/tvec으로 DJ 부스 박스를 프레임에 그린다."""
    w, d, h = box_size
    cx = (pattern_size[0] - 1) * square_size / 2 - w / 2
    cy = (pattern_size[1] - 1) * square_size / 2 - d / 2

    box_pts = np.float32([
        [cx,     cy,      0],
        [cx + w, cy,      0],
        [cx + w, cy + d,  0],
        [cx,     cy + d,  0],
        [cx,     cy,     -h],
        [cx + w, cy,     -h],
        [cx + w, cy + d, -h],
        [cx,     cy + d, -h],
    ])

    img_pts, _ = cv.projectPoints(box_pts, rvec, tvec, K, dist)
    img_pts = img_pts.reshape(-1, 2).astype(np.int32)

    out = frame.copy()

    # 옆면 4개 검정 채우기 (상단면은 AR 텍스처가 덮음)
    BLACK = (15, 15, 15)
    side_faces = [
        img_pts[[0, 1, 5, 4]],
        img_pts[[1, 2, 6, 5]],
        img_pts[[2, 3, 7, 6]],
        img_pts[[3, 0, 4, 7]],
    ]
    for face in side_faces:
        cv.fillPoly(out, [face], BLACK)

    # 윤곽선
    cv.drawContours(out, [img_pts[:4]], -1, (40, 40, 40),  1)
    for i in range(4):
        cv.line(out, tuple(img_pts[i]), tuple(img_pts[i + 4]), (40, 40, 40), 1)
    cv.drawContours(out, [img_pts[4:]], -1, (40, 40, 40),  1)
    return out


def make_mouse_callback(player, waveform_height=150, width=1000, pad=10):
    w = width - 2 * pad
    state = {'dragging': False, 'track': None, 'last_x': 0, 'last_t': 0.0}

    def callback(event, x, y, flags, param):
        track = 0 if y < waveform_height else 1

        if event == cv.EVENT_LBUTTONDOWN:
            state['dragging'] = True
            state['track']    = track
            state['last_x']   = x
            state['last_t']   = time.time()
            player.start_scratch(track)

        elif event == cv.EVENT_MOUSEMOVE and state['dragging']:
            now = time.time()
            dx  = x - state['last_x']
            dt  = now - state['last_t']
            state['last_x'] = x
            state['last_t'] = now
            delta_sec = dx * WINDOW_SEC / w * 0.04
            player.scratch(state['track'], delta_sec, dt)

        elif event == cv.EVENT_LBUTTONUP and state['dragging']:
            state['dragging'] = False
            player.end_scratch(state['track'])

    return callback


def main():
    left_path  = find_mp3("sound/left")
    right_path = find_mp3("sound/right")

    print("분석 중...")
    left_bpm,  left_amp,  left_offset  = analyze(left_path)
    right_bpm, right_amp, right_offset = analyze(right_path)
    print(f"LEFT : {left_path.name}  {left_bpm} BPM  offset={left_offset:.3f}s")
    print(f"RIGHT: {right_path.name}  {right_bpm} BPM  offset={right_offset:.3f}s")

    left_bar_dur  = 4 * 60 / left_bpm
    right_bar_dur = 4 * 60 / right_bpm
    player = MusicPlayer()
    player.load(0, left_path)
    player.load(1, right_path)

    cap0 = ThreadedCamera(0)
    cap1 = ThreadedCamera(1)
    cap2 = ThreadedCamera(2)
    if cap2._alive:
        CAM2_W = int(cap2._cap.get(cv.CAP_PROP_FRAME_WIDTH)  * 0.6)
        CAM2_H = int(cap2._cap.get(cv.CAP_PROP_FRAME_HEIGHT) * 0.6)
    else:
        CAM2_W = int(CAM_W * 0.4)
        CAM2_H = int(CAM_H * 0.4)

    cams, pose = load_pose_and_calib()
    if cams and pose:
        print("pose.txt / cali_result.txt 로드 완료 — 박스 오버레이 활성화")
    else:
        print("pose.txt 또는 cali_result.txt 없음 — 박스 오버레이 비활성화")

    _ht_raw = HandTracker()
    hand_tracker = AsyncHandTracker(_ht_raw)

    bar_durs = [left_bar_dur,  right_bar_dur]
    offsets  = [left_offset,   right_offset]

    jog_angles = [0.0, 0.0]
    jog_last_t = time.time()

    SCRATCH_Y_SCALE = 0.1
    scratch_st = [
        {'active': False, 'last_y': None, 'last_t': None},
        {'active': False, 'last_y': None, 'last_t': None},
    ]
    bot_prev = [{z: False for z in ('bot1', 'bot2', 'bot3')} for _ in range(2)]

    cv.namedWindow("Music Player")
    cv.setMouseCallback("Music Player", make_mouse_callback(player))

    topview_cache = draw_top_view([], active_states=[[False]*3, [False]*3])
    ar_H = [None, None]

    while True:
        player.update()

        left_img  = draw_waveform(
            left_amp,  left_bpm,  left_offset,
            f"[L] {left_path.name}  {left_bpm} BPM",
            pos=player.get_pos(0),
            loop=player.get_loop(0),
            cue=player.get_cue(0),
        )
        right_img = draw_waveform(
            right_amp, right_bpm, right_offset,
            f"[R] {right_path.name}  {right_bpm} BPM",
            pos=player.get_pos(1),
            loop=player.get_loop(1),
            cue=player.get_cue(1),
        )

        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if ret2 and frame2 is not None:
            frame2 = cv.flip(frame2, 1)

        # 핸드 트래킹은 박스 오버레이 전 원본 프레임으로 수행
        raw0 = frame0.copy() if ret0 and frame0 is not None else None
        raw1 = frame1.copy() if ret1 and frame1 is not None else None
        if raw0 is not None and raw1 is not None:
            hand_tracker.submit(raw0, raw1)

        # 원본 해상도에서 박스 옆면 채우기, 리사이즈 후 AR 합성
        if cams and pose:
            if ret0 and frame0 is not None:
                frame0 = draw_booth_box(
                    frame0,
                    cams['camera0']['matrix'], cams['camera0']['dist'],
                    pose['camera0']['rvec'],   pose['camera0']['tvec'],
                )
            if ret1 and frame1 is not None:
                frame1 = draw_booth_box(
                    frame1,
                    cams['camera1']['matrix'], cams['camera1']['dist'],
                    pose['camera1']['rvec'],   pose['camera1']['tvec'],
                )

        cam0 = cv.resize(frame0, (CAM_W, CAM_H)) if ret0 and frame0 is not None \
               else np.full((CAM_H, CAM_W, 3), 40, dtype=np.uint8)
        cam1 = cv.resize(frame1, (CAM_W, CAM_H)) if ret1 and frame1 is not None \
               else np.full((CAM_H, CAM_W, 3), 40, dtype=np.uint8)
        cam2_img = cv.resize(frame2, (CAM2_W, CAM2_H)) if ret2 and frame2 is not None \
                   else np.full((CAM2_H, CAM2_W, 3), 40, dtype=np.uint8)
        cv.putText(cam2_img, "CAM 3", (4, 18), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # H 최초 계산 (pose 고정이므로 1회만)
        if cams and pose and ar_H[0] is None and ret0 and frame0 is not None \
                and ret1 and frame1 is not None:
            oh0, ow0 = frame0.shape[:2]
            oh1, ow1 = frame1.shape[:2]
            ar_H[0] = compute_ar_H(
                cams['camera0']['matrix'], cams['camera0']['dist'],
                pose['camera0']['rvec'],   pose['camera0']['tvec'],
                ow0, oh0, CAM_W, CAM_H,
            )
            ar_H[1] = compute_ar_H(
                cams['camera1']['matrix'], cams['camera1']['dist'],
                pose['camera1']['rvec'],   pose['camera1']['tvec'],
                ow1, oh1, CAM_W, CAM_H,
            )

        if ar_H[0] is not None:
            cam0 = apply_ar_overlay(cam0, topview_cache, ar_H[0])
            cam1 = apply_ar_overlay(cam1, topview_cache, ar_H[1])

        cv.putText(cam0, "CAM 1", (8, 28), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv.putText(cam1, "CAM 2", (8, 28), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        side = (WAVEFORM_W - CAM_W * 2) // 2
        cam_row = np.hstack([
            np.full((CAM_H, side, 3), 20, dtype=np.uint8),
            cam0,
            cam1,
            np.full((CAM_H, WAVEFORM_W - CAM_W * 2 - side, 3), 20, dtype=np.uint8),
        ])
        main_panel = np.vstack([left_img, right_img, cam_row])


        hands0, hands1 = hand_tracker.get()
        finger_pts = []
        if cams and pose and raw0 is not None and raw1 is not None:
            finger_pts = _ht_raw.get_finger_pts(
                hands0, hands1, raw0.shape, raw1.shape, cams, pose
            )

        # ── 스크래치 연동 ─────────────────────────────────────
        now_t   = time.time()
        touched = [False, False]

        bot_cur = [{z: False for z in ('bot1', 'bot2', 'bot3')} for _ in range(2)]

        for pt in finger_pts:
            sec = get_section(pt)
            if sec is None:
                continue
            track, zone = sec
            clicking = float(pt[2]) > -1.0

            if zone == 'top' and clicking:
                touched[track] = True
                st = scratch_st[track]
                if not st['active']:
                    player.start_scratch(track)
                    st['active'] = True
                    st['last_y'] = float(pt[1])
                    st['last_t'] = now_t
                else:
                    dy = float(pt[1]) - st['last_y']
                    dt = now_t - st['last_t']
                    player.scratch(track, dy * SCRATCH_Y_SCALE, dt)
                    st['last_y'] = float(pt[1])
                    st['last_t'] = now_t

            elif zone in ('bot1', 'bot2', 'bot3') and clicking:
                bot_cur[track][zone] = True

        # 스크래치 해제
        for i in range(2):
            if not touched[i] and scratch_st[i]['active']:
                player.end_scratch(i)
                scratch_st[i]['active'] = False
                scratch_st[i]['last_y'] = None

        # 하단 구역 동작
        for i in range(2):
            # bot1: 재생/정지 — 누르는 순간 토글 (rising edge)
            if bot_cur[i]['bot1'] and not bot_prev[i]['bot1']:
                player.toggle(i)
            # bot2: 루프 on/off — 누르는 순간 토글 (rising edge)
            if bot_cur[i]['bot2'] and not bot_prev[i]['bot2']:
                player.toggle_loop(i, bar_durs[i] * 4, offsets[i])
            # bot3: 큐포인트 — 누르는 순간 현재 마디 첫박으로 설정
            if bot_cur[i]['bot3'] and not bot_prev[i]['bot3']:
                player.set_cue(i, bar_durs[i], offsets[i])

        bot_prev = bot_cur
        # ──────────────────────────────────────────────────────

        active_states = [
            [player.is_playing[i],
             player.get_loop(i) is not None,
             player.get_cue(i)  is not None]
            for i in range(2)
        ]
        now_jog    = time.time()
        dt_jog     = now_jog - jog_last_t
        jog_last_t = now_jog
        for i in range(2):
            if player.is_playing[i]:
                jog_angles[i] = (jog_angles[i] + 180.0 * dt_jog) % 360.0

        topview_cache = draw_top_view(finger_pts, active_states=active_states, angles=jog_angles)
        tv_h = topview_cache.shape[0]
        tv_cropped = topview_cache[int(tv_h * 0.1):int(tv_h * 0.9), :]
        topview_resized = cv.resize(tv_cropped, (CAM2_W, int(tv_cropped.shape[0] * CAM2_W / topview_cache.shape[1])))
        combined_view = np.vstack([cam2_img, topview_resized])
        target_h = main_panel.shape[0]
        scale = target_h / combined_view.shape[0]
        combined_scaled = cv.resize(combined_view,
                                    (int(combined_view.shape[1] * scale), target_h))
        cv.imshow("Music Player", np.hstack([combined_scaled, main_panel]))

        if cv.waitKey(30) & 0xFF == 27:
            break

    player.stop_all()
    cap0.release()
    cap1.release()
    cap2.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
