import cv2
import mediapipe as mp

# Inisialisasi MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Pose detector
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Webcam
cap = cv2.VideoCapture(0)

print("Program berjalan. Tekan tombol 'q' untuk keluar.")

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        print("Gagal membaca kamera.")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # Convert BGR ke RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Proses pose detection
    results = pose.process(image_rgb)

    # Kalau pose terdeteksi
    if results.pose_landmarks:

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

    # Tampilkan hasil
    cv2.imshow("MediaPipe Pose", frame)

    # Tekan q untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()