import cv2
import csv
import os
import mediapipe.python.solutions.pose as mp_pose

# Inisialisasi MediaPipe Pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ID Landmark yang kita gunakan (12 titik penting)
landmark_ids = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# Nama file CSV tempat menyimpan dataset
csv_filename = "dataset_latihan.csv"

# Jika file CSV belum ada, buat baru dan tulis Header-nya
if not os.path.exists(csv_filename):
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = []
        # Membuat header x1, y1, z1, x2, y2, z2 ... dst
        for i in range(len(landmark_ids)):
            header.extend([f'x{i+1}', f'y{i+1}', f'z{i+1}'])
        header.append('label') # Kolom terakhir untuk target/label
        writer.writerow(header)


cap = cv2.VideoCapture(0)



# Atur format videonya ke MJPEG biar sinkron sama DroidCam via USB
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

status_rekam = "IDLE (Tidak Merekam)"
label_aktif = -1

print("=== PROGRAM PENGUMPUL DATA DATASET ===")
print("Kontrol Keyboard:")
print("- Tekan 'p' : Mulai rekam gerakan PUSH-UP (Label 0)")
print("- Tekan 's' : Mulai rekam gerakan SQUAT (Label 1)")
print("- Tekan 'space' (Spasi) : Jeda/Berhenti merekam")
print("- Tekan 'q' : Keluar dari program\n")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    # Logika Perekaman Data
    if results.pose_landmarks and label_aktif != -1:
        baris_data = []
        landmarks = results.pose_landmarks.landmark
        
        # Ambil koordinat x, y, z dari 12 titik penting secara berurutan
        for idx in landmark_ids:
            lm = landmarks[idx]
            baris_data.extend([lm.x, lm.y, lm.z]) # Menggunakan koordinat ternormalisasi (0-1) agar lebih stabil untuk CNN
        
        # Tambahkan label di paling akhir baris
        baris_data.append(label_aktif)
        
        # Simpan ke file CSV
        with open(csv_filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(baris_data)
            
        print(f"Merekam... Label: {label_aktif} | Data: {len(baris_data)} kolom ditulis.")

    # Gambar teks status di layar kamera
    cv2.putText(frame, f"Status: {status_rekam}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Data Collector", frame)

    # Logika Keyboard Kontrol
    key = cv2.waitKey(1) & 0xFF
    if key == ord('p') or key == ord('P'):
        label_aktif = 0
        status_rekam = "MEREKAM PUSH-UP (0)"
    elif key == ord('s') or key == ord('S'):
        label_aktif = 1
        status_rekam = "MEREKAM SQUAT (1)"
    elif key == ord(' '): # Tombol Spasi
        label_aktif = -1
        status_rekam = "PAUSED / IDLE"
    elif key == ord('q') or key == ord('Q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Selesai! Dataset kamu tersimpan aman di file '{csv_filename}'")