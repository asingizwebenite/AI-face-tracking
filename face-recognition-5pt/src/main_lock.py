"""
Face Locking System with Multi-face Recognition
- Recognizes ALL enrolled people in the database
- Locks onto TARGET_IDENTITY when similarity threshold is met
- Only logs actions (blink, smile, head movement) for the locked person
- Shows both locked and recognized people with different colors
- Everything in one file - no external dependencies needed
"""

from __future__ import annotations
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

try:
    import mediapipe as mp
except Exception as e:
    mp = None
    _MP_IMPORT_ERROR = e

# Import align_face_5pt if available, otherwise define a placeholder
try:
    from .haar_5pt import align_face_5pt
except ImportError:
    # Define a simple fallback
    def align_face_5pt(image, landmarks, out_size=(112, 112)):
        """Simple alignment fallback - just crops the face"""
        x_coords = landmarks[:, 0]
        y_coords = landmarks[:, 1]
        x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
        y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
        
        # Expand bounding box
        width = x_max - x_min
        height = y_max - y_min
        x_min = max(0, x_min - width // 4)
        x_max = min(image.shape[1], x_max + width // 4)
        y_min = max(0, y_min - height // 4)
        y_max = min(image.shape[0], y_max + height // 4)
        
        face_crop = image[y_min:y_max, x_min:x_max]
        if face_crop.size == 0:
            return np.zeros((out_size[1], out_size[0], 3), dtype=np.uint8), np.eye(2, 3)
        
        aligned = cv2.resize(face_crop, out_size)
        return aligned, np.eye(2, 3)

# =========================
# 🔹 Configuration
# =========================
TARGET_IDENTITY = "benite"  # Person to lock onto
MAX_LOST_FRAMES = 10        # How many frames to wait before unlocking
LOCK_THRESHOLD = 0.6        # Minimum similarity for locking (0-1)
MOVEMENT_THRESHOLD = 30     # Pixels movement to trigger
BLINK_EAR_THRESHOLD = 0.25  # Eye Aspect Ratio threshold for blink
SMILE_WIDTH_THRESHOLD = 0.3 # Ratio threshold for smile

# =========================
# 🔹 Data Classes
# =========================

@dataclass
class FaceDet:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    kps: np.ndarray  # (5,2) float32 in FULL-frame coords

@dataclass
class MatchResult:
    name: Optional[str]
    distance: float
    similarity: float
    accepted: bool

@dataclass
class FaceResult:
    name: str
    similarity: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    landmarks: List[List[float]]     # 5-point landmarks
    is_locked: bool = False          # Whether this face is the locked one

# =========================
# 🔹 Action Detection Functions
# =========================

def eye_aspect_ratio(eye_points):
    """Calculate Eye Aspect Ratio for blink detection."""
    if len(eye_points) < 6:
        return 1.0
    
    # For 5-point landmarks, we need to approximate
    # Assuming eye_points are [left_corner, right_corner]
    if len(eye_points) == 2:
        # Simple approximation for 2-point eye landmarks
        eye_width = abs(eye_points[1][0] - eye_points[0][0])
        eye_height = abs(eye_points[1][1] - eye_points[0][1])
        return eye_height / max(eye_width, 1.0)
    
    # For 6-point eye landmarks
    A = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
    B = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
    C = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
    
    ear = (A + B) / (2.0 * C)
    return ear

def detect_blink(left_eye, right_eye, threshold=0.25):
    """Detect blink from eye landmarks."""
    # left_eye and right_eye are single points from 5-point landmarks
    # We need to approximate eye closure
    left_ear = eye_aspect_ratio([left_eye])
    right_ear = eye_aspect_ratio([right_eye])
    
    avg_ear = (left_ear + right_ear) / 2.0
    if avg_ear < threshold:
        return "BLINK"
    return None

def detect_head_movement(nose_x, threshold=30):
    """Detect head movement from nose position."""
    if not hasattr(detect_head_movement, 'last_nose_x'):
        detect_head_movement.last_nose_x = nose_x
        detect_head_movement.movement_count = 0
    
    movement = abs(nose_x - detect_head_movement.last_nose_x)
    detect_head_movement.last_nose_x = nose_x
    
    if movement > threshold:
        detect_head_movement.movement_count += 1
        if detect_head_movement.movement_count >= 2:
            detect_head_movement.movement_count = 0
            return "HEAD_MOVE"
    return None

def detect_smile(mouth_left, mouth_right, threshold=0.3):
    """Detect smile from mouth landmarks."""
    # mouth_left and mouth_right are single points
    mouth_width = abs(mouth_right[0] - mouth_left[0])
    
    if not hasattr(detect_smile, 'neutral_width'):
        detect_smile.neutral_width = mouth_width
    
    width_ratio = mouth_width / detect_smile.neutral_width
    if width_ratio > (1 + threshold):
        return "SMILE"
    return None

# =========================
# 🔹 History Logger
# =========================

class HistoryLogger:
    """Logs actions for a specific person."""
    def __init__(self, person_name: str):
        self.person_name = person_name
        self.log_file = Path(f"logs/{person_name}_actions.json")
        self.log_file.parent.mkdir(exist_ok=True)
        self.actions = []
        
    def log(self, action: str):
        """Log an action with timestamp."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        entry = {
            "timestamp": timestamp,
            "action": action,
            "person": self.person_name
        }
        self.actions.append(entry)
        print(f"[LOG] {timestamp} - {self.person_name}: {action}")
        
    def save(self):
        """Save logs to file."""
        if self.actions:
            with open(self.log_file, 'w') as f:
                json.dump(self.actions, f, indent=2)

# =========================
# 🔹 Face Lock System
# =========================

class FaceLock:
    """Manages locking onto a specific person."""
    def __init__(self, target_identity: str, max_lost_frames: int = 10):
        self.target_identity = target_identity
        self.max_lost_frames = max_lost_frames
        self.locked = False
        self.locked_id = None
        self.lost_frames = 0
        self.last_bbox = None
        
    def try_lock(self, name: str, similarity: float, threshold: float, bbox: Tuple) -> bool:
        """Try to lock onto a person if conditions are met."""
        if not self.locked and name == self.target_identity and similarity >= threshold:
            self.locked = True
            self.locked_id = name
            self.lost_frames = 0
            self.last_bbox = bbox
            print(f"[LOCK] Locked onto {name} with similarity {similarity:.3f}")
            return True
        return False
    
    def update_tracking(self, name: str, bbox: Tuple):
        """Update tracking state for locked person."""
        if self.locked and name == self.locked_id:
            self.lost_frames = 0
            self.last_bbox = bbox
        elif self.locked:
            self.lost_frames += 1
            if self.lost_frames >= self.max_lost_frames:
                print(f"[UNLOCK] Lost {self.locked_id} for {self.lost_frames} frames")
                self.locked = False
                self.locked_id = None
                self.lost_frames = 0

# =========================
# 🔹 Math Helpers
# =========================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    return float(np.dot(a, b))

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - cosine_similarity(a, b)

def _clip_xyxy(x1: float, y1: float, x2: float, y2: float, W: int, H: int) -> Tuple[int, int, int, int]:
    x1 = int(max(0, min(W - 1, round(x1))))
    y1 = int(max(0, min(H - 1, round(y1))))
    x2 = int(max(0, min(W - 1, round(x2))))
    y2 = int(max(0, min(H - 1, round(y2))))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2

def _bbox_from_5pt(kps: np.ndarray, pad_x: float = 0.55, pad_y_top: float = 0.85, pad_y_bot: float = 1.15) -> np.ndarray:
    """
    Build a nicer face-like bbox from 5 points with asymmetric padding.
    kps: (5,2) in full-frame coords
    """
    k = kps.astype(np.float32)
    x_min, x_max = float(np.min(k[:, 0])), float(np.max(k[:, 0]))
    y_min, y_max = float(np.min(k[:, 1])), float(np.max(k[:, 1]))

    w, h = max(1.0, x_max - x_min), max(1.0, y_max - y_min)

    x1 = x_min - pad_x * w
    x2 = x_max + pad_x * w
    y1 = y_min - pad_y_top * h
    y2 = y_max + pad_y_bot * h
    return np.array([x1, y1, x2, y2], dtype=np.float32)

def _kps_span_ok(kps: np.ndarray, min_eye_dist: float) -> bool:
    """
    Minimal geometry sanity:
    - eyes not collapsed
    - mouth generally below nose
    """
    k = kps.astype(np.float32)
    le, re, no, lm, rm = k
    eye_dist = float(np.linalg.norm(re - le))
    if eye_dist < float(min_eye_dist):
        return False
    if not (lm[1] > no[1] and rm[1] > no[1]):
        return False
    return True

# =========================
# 🔹 DB Helpers
# =========================

def load_db_npz(db_path: Path) -> Dict[str, np.ndarray]:
    if not db_path.exists():
        return {}
    data = np.load(str(db_path), allow_pickle=True)
    out: Dict[str, np.ndarray] = {}
    for k in data.files:
        out[k] = np.asarray(data[k], dtype=np.float32).reshape(-1)
    return out

# =========================
# 🔹 Embedder
# =========================

class ArcFaceEmbedderONNX:
    """
    ArcFace-style ONNX embedder.
    Input: 112x112 BGR -> internally RGB + (x-127.5)/128, NCHW float32
    Output: (D,)
    """
    def __init__(self, model_path: str = "models/embedder_arcface.onnx",
                 input_size: Tuple[int,int] = (112,112),
                 debug: bool = False):
        self.model_path = model_path
        self.in_w, self.in_h = int(input_size[0]), int(input_size[1])
        self.debug = bool(debug)

        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name

        if self.debug:
            print("[embed] model:", model_path)
            print("[embed] input:", self.in_name, self.sess.get_inputs()[0].shape)
            print("[embed] output:", self.out_name, self.sess.get_outputs()[0].shape)

    def _preprocess(self, aligned_bgr_112: np.ndarray) -> np.ndarray:
        img = aligned_bgr_112
        if img.shape[1] != self.in_w or img.shape[0] != self.in_h:
            img = cv2.resize(img, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 128.0
        x = rgb[None, ...]
        return x.astype(np.float32)

    @staticmethod
    def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        v = v.astype(np.float32).reshape(-1)
        n = float(np.linalg.norm(v) + eps)
        return (v / n).astype(np.float32)

    def embed(self, aligned_bgr_112: np.ndarray) -> np.ndarray:
        x = self._preprocess(aligned_bgr_112)
        y = self.sess.run([self.out_name], {self.in_name: x})[0]
        emb = np.asarray(y, dtype=np.float32).reshape(-1)
        return self._l2_normalize(emb)

# =========================
# 🔹 Multi-face Haar + FaceMesh(ROI) 5pt
# =========================

class HaarFaceMesh5pt:
    def __init__(self, haar_xml: Optional[str] = None,
                 min_size: Tuple[int,int] = (70,70),
                 debug: bool = False):
        self.debug = bool(debug)
        self.min_size = tuple(map(int, min_size))

        if haar_xml is None:
            haar_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(haar_xml)
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade: {haar_xml}")

        if mp is None:
            raise RuntimeError(f"mediapipe import failed: {_MP_IMPORT_ERROR}")

        # FaceMesh ROI
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # 5pt indices
        self.IDX_LEFT_EYE = 33
        self.IDX_RIGHT_EYE = 263
        self.IDX_NOSE_TIP = 1
        self.IDX_MOUTH_LEFT = 61
        self.IDX_MOUTH_RIGHT = 291

    def _haar_faces(self, gray: np.ndarray) -> np.ndarray:
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=self.min_size,
        )
        if faces is None or len(faces) == 0:
            return np.zeros((0, 4), dtype=np.int32)
        return faces.astype(np.int32)

    def _roi_facemesh_5pt(self, roi_bgr: np.ndarray) -> Optional[np.ndarray]:
        H, W = roi_bgr.shape[:2]
        if H < 20 or W < 20:
            return None
        rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None

        lm = res.multi_face_landmarks[0].landmark
        idxs = [self.IDX_LEFT_EYE, self.IDX_RIGHT_EYE, self.IDX_NOSE_TIP,
                self.IDX_MOUTH_LEFT, self.IDX_MOUTH_RIGHT]

        pts = []
        for i in idxs:
            p = lm[i]
            pts.append([p.x * W, p.y * H])

        kps = np.array(pts, dtype=np.float32)

        # enforce left/right ordering
        if kps[0,0] > kps[1,0]:
            kps[[0,1]] = kps[[1,0]]
        if kps[3,0] > kps[4,0]:
            kps[[3,4]] = kps[[4,3]]

        return kps

    def detect(self, frame_bgr: np.ndarray, max_faces: int = 5) -> List[FaceDet]:
        H, W = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        faces = self._haar_faces(gray)
        if faces.shape[0] == 0:
            return []

        # sort by area desc, keep top max_faces
        areas = faces[:,2]*faces[:,3]
        order = np.argsort(areas)[::-1]
        faces = faces[order][:max_faces]

        out: List[FaceDet] = []

        for (x,y,w,h) in faces:
            # expand ROI
            mx, my = 0.25*w, 0.35*h
            rx1, ry1, rx2, ry2 = _clip_xyxy(x-mx, y-my, x+w+mx, y+h+my, W, H)
            roi = frame_bgr[ry1:ry2, rx1:rx2]

            kps_roi = self._roi_facemesh_5pt(roi)
            if kps_roi is None:
                if self.debug:
                    print("[recognize] FaceMesh none for ROI -> skip")
                continue

            kps = kps_roi.copy()
            kps[:,0] += float(rx1)
            kps[:,1] += float(ry1)

            if not _kps_span_ok(kps, min_eye_dist=max(10.0, 0.18*float(w))):
                if self.debug:
                    print("[recognize] 5pt geometry failed -> skip")
                continue

            bb = _bbox_from_5pt(kps, pad_x=0.55, pad_y_top=0.85, pad_y_bot=1.15)
            x1, y1, x2, y2 = _clip_xyxy(bb[0], bb[1], bb[2], bb[3], W, H)

            out.append(FaceDet(x1=x1, y1=y1, x2=x2, y2=y2, score=1.0, kps=kps.astype(np.float32)))

        return out

# =========================
# 🔹 Matcher
# =========================

class FaceDBMatcher:
    def __init__(self, db: Dict[str,np.ndarray], dist_thresh: float = 0.34):
        self.db = db
        self.dist_thresh = float(dist_thresh)
        self._names: List[str] = []
        self._mat: Optional[np.ndarray] = None
        self._rebuild()

    def _rebuild(self):
        self._names = sorted(self.db.keys())
        if self._names:
            self._mat = np.stack([self.db[n].reshape(-1).astype(np.float32) for n in self._names], axis=0)
        else:
            self._mat = None

    def reload_from(self, path: Path):
        self.db = load_db_npz(path)
        self._rebuild()

    def match(self, emb: np.ndarray) -> MatchResult:
        if self._mat is None or len(self._names) == 0:
            return MatchResult(name=None, distance=1.0, similarity=0.0, accepted=False)

        e = emb.reshape(1,-1).astype(np.float32)
        sims = (self._mat @ e.T).reshape(-1)
        best_i = int(np.argmax(sims))
        best_sim = float(sims[best_i])
        best_dist = 1.0 - best_sim
        ok = best_dist <= self.dist_thresh

        return MatchResult(
            name=self._names[best_i] if ok else None,
            distance=float(best_dist),
            similarity=float(best_sim),
            accepted=bool(ok)
        )

# =========================
# 🔹 Complete Face Locking System
# =========================

class FaceLockingSystem:
    """Complete system that recognizes all faces, locks onto one, and logs actions."""
    
    def __init__(self, db_path="data/db/face_db.npz", camera_id=0):
        self.db_path = Path(db_path)
        self.camera_id = camera_id
        
        # Initialize face recognition components
        self.det = HaarFaceMesh5pt(min_size=(70, 70), debug=False)
        self.embedder = ArcFaceEmbedderONNX(
            model_path="models/embedder_arcface.onnx",
            input_size=(112, 112)
        )
        
        # Load database
        db = load_db_npz(self.db_path)
        self.matcher = FaceDBMatcher(db=db, dist_thresh=0.34)
        
        # Initialize face lock system
        self.face_lock = FaceLock(TARGET_IDENTITY, MAX_LOST_FRAMES)
        self.logger = None
        
        # Camera
        self.cap = None
        
        # Statistics
        self.frame_count = 0
        self.start_time = time.time()
        
    def initialize_camera(self):
        """Initialize the camera."""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Camera {self.camera_id} not available")
        
        print(f"Camera initialized (ID: {self.camera_id})")
        print(f"Database loaded: {len(self.matcher._names)} identities")
        print(f"Target to lock: {TARGET_IDENTITY}")
        print(f"Lock threshold: {LOCK_THRESHOLD}")
        print("=" * 50)
    
    def recognize_faces(self, frame_bgr):
        """
        Recognize all faces in a frame.
        Returns list of FaceResult objects.
        """
        # Detect faces
        faces = self.det.detect(frame_bgr, max_faces=5)
        results = []
        
        for f in faces:
            # Align and embed
            aligned, _ = align_face_5pt(frame_bgr, f.kps, out_size=(112, 112))
            emb = self.embedder.embed(aligned)
            
            # Match with database
            mr = self.matcher.match(emb)
            name = mr.name if mr.name else "Unknown"
            
            # Check if this is the locked person
            is_locked = (self.face_lock.locked and name == self.face_lock.locked_id)
            
            # Create result
            result = FaceResult(
                name=name,
                similarity=float(mr.similarity),
                bbox=(f.x1, f.y1, f.x2, f.y2),
                landmarks=f.kps.tolist(),
                is_locked=is_locked
            )
            results.append(result)
        
        return results
    
    def process_frame(self, frame):
        """
        Process a frame: recognize faces, update lock, detect actions.
        Returns the frame with visualization.
        """
        self.frame_count += 1
        
        # Recognize all faces
        face_results = self.recognize_faces(frame)
        
        # Process each face
        for result in face_results:
            name = result.name
            similarity = result.similarity
            bbox = result.bbox
            landmarks = result.landmarks
            
            # Try to lock if this is the target person
            locked_now = self.face_lock.try_lock(name, similarity, LOCK_THRESHOLD, bbox)
            
            # Initialize logger if locked now
            if locked_now and self.logger is None:
                self.logger = HistoryLogger(name)
                print(f"[SYSTEM] Started logging for {name}")
            
            # Update tracking
            self.face_lock.update_tracking(name, bbox)
            
            # Draw visualization
            color = (0, 255, 0) if result.is_locked else (255, 0, 0)  # Green for locked, blue for others
            thickness = 3 if result.is_locked else 2
            
            # Draw bounding box
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw landmarks (small dots)
            for (x, y) in landmarks:
                cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
            
            # Draw label
            label = f"{name} ({similarity:.2f})"
            if result.is_locked:
                label += " [LOCKED]"
            
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Detect and log actions for locked person only
            if result.is_locked:
                left_eye, right_eye, nose, mouth_left, mouth_right = landmarks
                
                # Detect actions
                move = detect_head_movement(nose[0], MOVEMENT_THRESHOLD)
                blink = detect_blink(left_eye, right_eye, BLINK_EAR_THRESHOLD)
                smile = detect_smile(mouth_left, mouth_right, SMILE_WIDTH_THRESHOLD)
                
                # Log actions
                for action in [move, blink, smile]:
                    if action:
                        print(f"[ACTION] {name}: {action}")
                        if self.logger:
                            self.logger.log(action)
        
        # Draw system info
        fps = self.frame_count / (time.time() - self.start_time) if (time.time() - self.start_time) > 0 else 0
        
        info_lines = [
            f"FPS: {fps:.1f}",
            f"Faces: {len(face_results)}",
            f"Locked: {self.face_lock.locked_id or 'None'}",
            f"DB Size: {len(self.matcher._names)}"
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (10, 30 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw "LOCKED" indicator if someone is locked
        if self.face_lock.locked:
            cv2.putText(frame, "LOCKED", (frame.shape[1] - 120, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame
    
    def run(self):
        """Main loop to run the face locking system."""
        self.initialize_camera()
        
        print("Face Locking System Started")
        print("Press 'q' to quit")
        print("Press 'r' to reload database")
        print("Press 's' to save logs")
        print("=" * 50)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Show frame
                cv2.imshow("Face Locking System", processed_frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Reload database
                    db = load_db_npz(self.db_path)
                    self.matcher = FaceDBMatcher(db=db, dist_thresh=0.34)
                    print(f"[SYSTEM] Database reloaded: {len(self.matcher._names)} identities")
                elif key == ord('s'):
                    # Save logs
                    if self.logger:
                        self.logger.save()
                        print(f"[SYSTEM] Logs saved for {self.logger.person_name}")
                    else:
                        print("[SYSTEM] No logs to save (no one is locked)")
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            
            # Save logs if any
            if self.logger:
                self.logger.save()
                print(f"[SYSTEM] Logs saved for {self.logger.person_name}")
            
            print(f"\nSystem ran for {self.frame_count} frames")
            print(f"Average FPS: {self.frame_count / (time.time() - self.start_time):.1f}")
            print("System shutdown complete")

# =========================
# 🔹 Main Function
# =========================

def main():
    """Main entry point."""
    system = FaceLockingSystem(db_path="data/db/face_db.npz", camera_id=0)
    system.run()

# =========================
# 🔹 Backward Compatible API
# =========================

def recognize_faces(frame_bgr, db_path="data/db/face_db.npz"):
    """
    Backward compatible API function.
    Returns a list of face dictionaries for Face Locking system.
    """
    if not hasattr(recognize_faces, "system"):
        recognize_faces.system = FaceLockingSystem(db_path, camera_id=0)
        # Initialize without camera
        db = load_db_npz(Path(db_path))
        recognize_faces.system.matcher = FaceDBMatcher(db=db, dist_thresh=0.34)
    
    # Use the recognition part of the system
    faces = recognize_faces.system.det.detect(frame_bgr, max_faces=5)
    results = []
    
    for f in faces:
        aligned, _ = align_face_5pt(frame_bgr, f.kps, out_size=(112, 112))
        emb = recognize_faces.system.embedder.embed(aligned)
        mr = recognize_faces.system.matcher.match(emb)
        
        name = mr.name if mr.name else "Unknown"
        
        results.append({
            "name": name,
            "similarity": float(mr.similarity),
            "bbox": (f.x1, f.y1, f.x2, f.y2),
            "landmarks": f.kps.tolist()
        })
    
    return results

# =========================
# 🔹 Run as Script
# =========================

if __name__ == "__main__":
    main()