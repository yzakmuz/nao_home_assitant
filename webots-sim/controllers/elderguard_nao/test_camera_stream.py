"""
test_camera_stream.py — Test the Webots camera stream.

Connects to the Webots NAO controller's camera server (port 5556),
receives frames, and displays them in an OpenCV window. Shows FPS
and frame resolution.

Run this while the Webots simulation is running.

Usage:
    python test_camera_stream.py [--port 5556] [--no-display]
"""

import sys
import time

# Add current directory for webots_camera import
sys.path.insert(0, ".")

from webots_camera import WebotsCamera

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def main():
    port = 5556
    no_display = "--no-display" in sys.argv
    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        port = int(sys.argv[idx + 1])

    print("=" * 50)
    print("  Webots Camera Stream Test")
    print("  Connecting to 127.0.0.1:%d ..." % port)
    print("=" * 50)

    cam = WebotsCamera("127.0.0.1", port)
    if not cam.start():
        print("[ERROR] Could not connect to camera server.")
        print("        Is the Webots simulation running?")
        sys.exit(1)

    print("[OK] Connected. Receiving frames...")
    if no_display or not HAS_CV2:
        if not HAS_CV2:
            print("[INFO] OpenCV not available — headless mode")
        print("[INFO] Headless mode — press Ctrl+C to stop")
    else:
        print("[INFO] Press 'q' in the window to stop")

    frame_count = 0
    start_time = time.monotonic()

    try:
        while True:
            frame = cam.read()
            if frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            elapsed = time.monotonic() - start_time

            if no_display or not HAS_CV2:
                # Headless: just print stats every second
                if frame_count % 15 == 0:
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    print("  Frame %d  |  %dx%d  |  %.1f FPS" % (
                        frame_count, frame.shape[1], frame.shape[0], fps))
            else:
                # Draw FPS overlay on frame
                fps = frame_count / elapsed if elapsed > 0 else 0
                label = "FPS: %.1f  |  %dx%d  |  Frame %d" % (
                    fps, frame.shape[1], frame.shape[0], frame_count)
                cv2.putText(frame, label, (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 1)

                cv2.imshow("Webots NAO Camera", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

            if not cam.is_running:
                print("[INFO] Camera connection lost.")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")

    total_time = time.monotonic() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    print("\n" + "=" * 50)
    print("  Total frames: %d" % frame_count)
    print("  Duration:     %.1f s" % total_time)
    print("  Average FPS:  %.1f" % avg_fps)
    print("=" * 50)

    cam.stop()
    if HAS_CV2:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
