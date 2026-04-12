import cv2
import numpy as np
import threading
import time

# =========================================================
# CONFIGURATION
# =========================================================
RTSP_URL = "rtsp://192.168.1.9:8554/stream"

# Canny edge thresholds
CANNY_LOW = 50
CANNY_HIGH = 150

# Edge overlay color (BGR) — neon cyan
EDGE_COLOR = (0, 255, 255)

# Shape detection thresholds
HOUGH_LINE_THRESHOLD = 80       # votes needed to keep a line
HOUGH_LINE_MIN_LEN   = 60       # minimum line segment length (px)
HOUGH_LINE_MAX_GAP   = 10       # max gap between segments to join

POLY_AREA_MIN = 500             # discard tiny contours

# Circle / ellipse detection (contour-based)
CIRCULARITY_THRESH  = 0.72      # 4π·area/perimeter² ≥ this → treat as round shape
ELLIPSE_AXIS_RATIO  = 0.85      # minor/major ≥ this → "Circle", else "Ellipse"

# High-sided polygon vs circle disambiguation
# For a regular n-gon the interior angle = (n-2)*180/n:
#   octagon → 135°, decagon → 144°, 12-gon → 150°
# A circle approximated by approxPolyDP has near-flat "corners" (≈ 170–178°).
# If the *minimum* corner angle across all vertices is below this threshold the
# shape is treated as a real polygon; otherwise it is classified as a circle/ellipse.
POLYGON_MAX_CORNER_ANGLE = 160  # degrees

# Colors for shape annotations (BGR)
COLOR_LINE    = (0, 0, 255)     # red
COLOR_CIRCLE  = (255, 0, 0)     # blue
COLOR_POLY    = (0, 255, 0)     # green
COLOR_LABEL   = (255, 255, 0)   # yellow


# =========================================================
# RTSP STREAM
# =========================================================
class RTSPStream:
    def __init__(self, url: str, reconnect: bool = True):
        self.url = url
        self.reconnect = reconnect
        self._cap = None
        self._frame = None
        self._running = False
        self._lock = threading.Lock()
        self._thread = None

    def _open_capture(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise ConnectionError(f"Could not connect to RTSP stream: {self.url}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return cap

    def start(self):
        self._cap = self._open_capture()
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return self

    def _capture_loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                if self.reconnect and self._running:
                    print("[INFO] Stream lost, reconnecting...")
                    try:
                        self._cap.release()
                    except Exception:
                        pass
                    time.sleep(1)
                    try:
                        self._cap = self._open_capture()
                    except ConnectionError:
                        continue
                continue
            with self._lock:
                self._frame = frame

    def read(self):
        with self._lock:
            if self._frame is None:
                return None
            frame = self._frame.copy()
        return cv2.resize(frame, (640, 480))

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._cap is not None:
            self._cap.release()

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# =========================================================
# PROCESSING
# =========================================================
def build_edge_overlay(frame):
    """Grayscale frame with Canny edges drawn in EDGE_COLOR."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)

    # Convert gray to BGR so we can draw in colour
    display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    display[edges != 0] = EDGE_COLOR
    return display


def min_corner_angle(approx):
    """Return the minimum interior angle (degrees) at any vertex of the polygon.

    Real polygon corners are sharp (well below 180°).
    Points sampled from a smooth circle curve are near-flat (≈ 170–178°).
    """
    pts = approx[:, 0, :].astype(float)
    n = len(pts)
    min_angle = 180.0
    for i in range(n):
        a = pts[(i - 1) % n] - pts[i]
        b = pts[(i + 1) % n] - pts[i]
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            continue
        cos_val = np.clip(np.dot(a, b) / denom, -1.0, 1.0)
        min_angle = min(min_angle, np.degrees(np.arccos(cos_val)))
    return min_angle


def poly_label(n):
    return {3: "Triangle", 4: "Quadrilateral", 5: "Pentagon",
            6: "Hexagon", 7: "Heptagon", 8: "Octagon",
            9: "Nonagon", 10: "Decagon"}.get(n, f"{n}-gon")


def build_shape_overlay(frame):
    """Grayscale frame with detected lines, circles, ellipses and polygons annotated."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # --- Lines (HoughLinesP) ---
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_LINE_THRESHOLD,
        minLineLength=HOUGH_LINE_MIN_LEN,
        maxLineGap=HOUGH_LINE_MAX_GAP
    )
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(display, (x1, y1), (x2, y2), COLOR_LINE, 2)

    # --- Circles, Ellipses and Polygons ---
    # RETR_TREE gives the full parent/child hierarchy so we can:
    #   (a) detect shapes nested inside other shapes, and
    #   (b) suppress duplicate labels (e.g. both edges of a ring → same label).
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return display

    # Pass 1 — classify every contour; store result by index.
    # Each entry is (label, draw_kind, draw_arg) or None if the contour is skipped.
    classifications = {}
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < POLY_AREA_MIN:
            classifications[i] = None
            continue
        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            classifications[i] = None
            continue

        circularity = 4 * np.pi * area / (peri * peri)

        # Coarse approximation for polygon classification.
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        n = len(approx)

        # Fine approximation to distinguish circles from high-sided polygons.
        # Real polygon sides are already straight, so vertex count stays the same
        # regardless of epsilon.  A circle's curved edges produce many more vertices
        # at fine tolerance (≈13–16 for 0.005·peri) while an octagon stays at 8.
        approx_fine = cv2.approxPolyDP(cnt, 0.005 * peri, True)
        n_fine = len(approx_fine)

        is_round = (
            circularity >= CIRCULARITY_THRESH
            and len(cnt) >= 5
            and n_fine > 12
        )

        if is_round:
            ellipse = cv2.fitEllipse(cnt)
            (_, _), (major_ax, minor_ax), _ = ellipse
            axis_ratio = min(major_ax, minor_ax) / max(major_ax, minor_ax) if max(major_ax, minor_ax) > 0 else 0
            label = "Circle" if axis_ratio >= ELLIPSE_AXIS_RATIO else "Ellipse"
            classifications[i] = (label, "ellipse", (ellipse, cnt))
        elif 3 <= n <= 12:
            classifications[i] = (poly_label(n), "poly", approx)
        else:
            classifications[i] = None

    # Pass 2 — draw, suppressing any contour whose parent already carries the
    # same label.  This removes the duplicate inner-edge label on rings while
    # keeping differently-labelled inner shapes (e.g. triangle inside a circle).
    for i, cnt in enumerate(contours):
        entry = classifications.get(i)
        if entry is None:
            continue

        label, kind, draw_arg = entry

        parent_idx = hierarchy[0][i][3]  # -1 when there is no parent
        if parent_idx != -1:
            parent_entry = classifications.get(parent_idx)
            if parent_entry is not None and parent_entry[0] == label:
                continue  # duplicate of parent — skip

        if kind == "ellipse":
            ellipse, cnt_orig = draw_arg
            cv2.ellipse(display, ellipse, COLOR_CIRCLE, 2)
            x, y, w, h = cv2.boundingRect(cnt_orig)
            cv2.putText(display, label, (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_LABEL, 1)
        else:  # poly
            approx = draw_arg
            cv2.drawContours(display, [approx], -1, COLOR_POLY, 2)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.putText(display, label, (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_LABEL, 1)

    return display


# =========================================================
# MAIN
# =========================================================
def main():
    print("Select view mode:")
    print("  1 - Regular (original color)")
    print("  2 - Edge detection")
    print("  3 - Shape detection")

    while True:
        choice = input("Enter 1, 2 or 3: ").strip()
        if choice in ("1", "2", "3"):
            break
        print("Invalid input, please enter 1, 2 or 3.")

    mode = int(choice)
    window_titles = {1: "Original", 2: "Edges", 3: "Shape Detection"}
    window_name = window_titles[mode]

    print(f"[INFO] Connecting to {RTSP_URL}")
    print("[INFO] Press q to quit")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        with RTSPStream(RTSP_URL) as stream:
            resolution_printed = False
            while True:
                frame = stream.read()
                if frame is None:
                    continue

                if not resolution_printed:
                    h, w = frame.shape[:2]
                    print(f"[INFO] Frame resolution: {w}x{h}")
                    resolution_printed = True

                if mode == 1:
                    display = frame
                elif mode == 2:
                    display = build_edge_overlay(frame)
                else:
                    display = build_shape_overlay(frame)

                cv2.imshow(window_name, display)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except ConnectionError as e:
        print(f"[ERROR] {e}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()