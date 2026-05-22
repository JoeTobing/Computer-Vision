import cv2
import mediapipe as mp

# =========================
# INISIALISASI MEDIAPIPE
# =========================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================
# DAFTAR NAMA LANDMARK YANG INGIN DIPAKAI
# =========================
# Kamu bisa hapus ID 0 (Nose) jika wajah benar-benar ingin diabaikan total
landmark_names = {
    11: "Left Shoulder", 12: "Right Shoulder",
    13: "Left Elbow",    14: "Right Elbow",
    15: "Left Wrist",    16: "Right Wrist",
    23: "Left Hip",      24: "Right Hip",
    25: "Left Knee",     26: "Right Knee",
    27: "Left Ankle",    28: "Right Ankle"
}

# =========================
# WEBCAM
# =========================
cap = cv2.VideoCapture(0)

print("Program berjalan...")
print("Tekan tombol Q untuk keluar")

# =========================
# LOOPING CAMERA
# =========================
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Kamera tidak terbaca")
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        # NOTE: mp_drawing bawaan akan tetap menggambar garis skeleton tipis secara global.
        # Jika kamu merasa terganggu dengan garis wajah, biarkan saja karena itu tidak memakan resource data koordinatmu.
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1)
        )

        # Ambil landmark
        for id, lm in enumerate(results.pose_landmarks.landmark):
            
            # --- DI SINI KUNCI FILTERNYA ---
            # Jika ID titik TIDAK ADA di dalam list penting kita, langsung dilewati (skip)
            if id not in landmark_names:
                continue 

            # Koordinat ini hanya diproses untuk titik yang lolos filter di atas
            cx = int(lm.x * w)
            cy = int(lm.y * h)

            # Gambar lingkaran biru besar khusus di titik penting
            cv2.circle(frame, (cx, cy), 6, (255, 0, 0), -1)

            # Tampilkan ID dan Nama Landmark pilihan
            info_teks = f"{id}: {landmark_names[id]}"
            cv2.putText(
                frame, info_teks, (cx + 10, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1
            )

            # Print ke terminal (Hanya memunculkan koordinat penting untuk modal dataset CSV nanti)
            print(f"{landmark_names[id]} ({id}) -> X:{cx} Y:{cy}")

    cv2.imshow("Pose Detection - Filtered", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()