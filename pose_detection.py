import cv2
import mediapipe as mp

# =========================
# INISIALISASI MEDIAPIPE
# =========================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Pose detector
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================
# DAFTAR NAMA LANDMARK
# =========================
landmark_names = {

    0: "Nose",

    11: "Left Shoulder",
    12: "Right Shoulder",

    13: "Left Elbow",
    14: "Right Elbow",

    15: "Left Wrist",
    16: "Right Wrist",

    23: "Left Hip",
    24: "Right Hip",

    25: "Left Knee",
    26: "Right Knee",

    27: "Left Ankle",
    28: "Right Ankle"
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

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # Ukuran frame
    h, w, c = frame.shape

    # Convert BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process pose
    results = pose.process(rgb)

    # Jika pose terdeteksi
    if results.pose_landmarks:

        # Gambar skeleton
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,

            mp_drawing.DrawingSpec(
                color=(0, 255, 0),
                thickness=2,
                circle_radius=2
            ),

            mp_drawing.DrawingSpec(
                color=(0, 0, 255),
                thickness=2,
                circle_radius=2
            )
        )

        # Ambil landmark
        for id, lm in enumerate(results.pose_landmarks.landmark):

            # Konversi koordinat
            cx = int(lm.x * w)
            cy = int(lm.y * h)

            # Gambar titik
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

            # Tampilkan ID
            cv2.putText(
                frame,
                str(id),
                (cx, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

            # Jika landmark penting
            if id in landmark_names:

                # Tampilkan nama landmark
                cv2.putText(
                    frame,
                    landmark_names[id],
                    (cx + 10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2
                )

                # Print koordinat
                print(f"{landmark_names[id]} -> X:{cx} Y:{cy}")

    # Tampilkan window
    cv2.imshow("Pose Detection", frame)

    # Tombol keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# RELEASE
# =========================
cap.release()
cv2.destroyAllWindows()
