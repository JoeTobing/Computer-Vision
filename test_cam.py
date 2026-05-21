import cv2
import numpy as np
import mediapipe.python.solutions.pose as mp_pose
import mediapipe.python.solutions.drawing_utils as mp_drawing

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS

cap = cv2.VideoCapture(0)

# Skenario Tambahan: Mengatur resolusi kamera agar lebih ringan dan responsif (FPS naik)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n=== PROGRAM BERJALAN ===")
print("PENTING: Klik dulu jendela videonya, baru tekan tombol untuk keluar!")
print("- Tekan tombol 'q' atau 'ESC' pada keyboard untuk keluar.")
print("- Atau langsung klik tombol silang (X) pada jendela video.\n")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Gagal mengambil gambar dari kamera.")
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, 
            results.pose_landmarks, 
            POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        )

    # Tampilkan ke layar
    cv2.imshow('Test MediaPipe Python 3.13', frame)

    # PERBAIKAN LOGIKA KEYBOARD (Lebih sensitif & Multi-opsi)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q') or key == 27: # 27 adalah kode biner untuk tombol ESC
        print("Program dihentikan melalui keyboard.")
        break
        
    # PERBAIKAN LOGIKA JENDELA (Jika tombol X diklik, program langsung mati)
    if cv2.getWindowProperty('Test MediaPipe Python 3.13', cv2.WND_PROP_VISIBLE) < 1:
        print("Program dihentikan karena jendela ditutup.")
        break

cap.release()
cv2.destroyAllWindows()
print("=== PROGRAM SELESAI ===")