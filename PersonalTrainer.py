import cv2
import mediapipe as mp
import numpy as np
import pickle

# 1. Load Otak AI yang baru saja kamu train
with open('model_trainer.pkl', 'rb') as f:
    model = pickle.load(f)

# 2. Inisialisasi MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# 3. Variabel untuk Hitung Repetisi (Reps Counter)
counter = 0
current_stage = None  # Untuk mendeteksi posisi 'down' atau 'up'

# ID Titik Sendi yang kita pakai kemarin (Harus sama urutannya!)
landmark_ids = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# 4. Nyalakan Webcam Laptop (Indeks 0)
cap = cv2.VideoCapture(0)

print("🚀 PERSONAL TRAINER AI AKTIF! Berdirilah di depan kamera...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Gagal mengambil gambar dari webcam.")
        break
    
    # Flip gambar biar kayak cermin + Ubah warna ke RGB
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    status_prediksi = "Mencari Tubuh..."
    warna_box = (0, 255, 255) # Kuning default
    
    if results.pose_landmarks:
        # Gambar titik robot di layar biar keren!
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Ambil koordinat x, y, z
        baris_data = []
        landmarks = results.pose_landmarks.landmark
        for idx in landmark_ids:
            lm = landmarks[idx]
            baris_data.extend([lm.x, lm.y, lm.z])
        
        # --- DISURUH MENEBAK OLEH MODEL PKL ---
        prediksi = model.predict([baris_data])[0]
        probabilitas = model.predict_proba([baris_data])[0]
        confidence = max(probabilitas) * 100
        
        # Logika Penentuan Gerakan & Hitung Repetisi
        if confidence > 80: # Hanya proses jika AI yakin di atas 80%
            if prediksi == 0:
                status_prediksi = f"PUSH-UP ({confidence:.0f}%)"
                warna_box = (0, 255, 0) # Hijau
                
                # Contoh logika counter Push-up sederhana berdasarkan koordinat hidung/bahu (titik 11)
                # Jika bahu di bawah setengah layar, dianggap 'down'
                bahu_y = landmarks[11].y
                if bahu_y > 0.6:
                    current_stage = "down"
                elif bahu_y < 0.45 and current_stage == "down":
                    current_stage = "up"
                    counter += 1
                    print(f"🔥 Push-up Berhasil! Total: {counter}")
                    
            elif prediksi == 1:
                status_prediksi = f"SQUAT ({confidence:.0f}%)"
                warna_box = (255, 0, 0) # Biru
                
                # Contoh logika counter Squat berdasarkan koordinat pinggul (titik 23)
                pinggul_y = landmarks[23].y
                if pinggul_y > 0.65:
                    current_stage = "down"
                elif pinggul_y < 0.55 and current_stage == "down":
                    current_stage = "up"
                    counter += 1
                    print(f"🔥 Squat Berhasil! Total: {counter}")
        else:
            status_prediksi = "Gerakan Kurang Jelas..."
            warna_box = (0, 0, 255) # Merah

    # 5. Desain Tampilan HUD ala Game di Layar Kamera
    # Box Status Gerakan
    cv2.rectangle(frame, (10, 10), (320, 60), warna_box, cv2.FILLED)
    cv2.putText(frame, status_prediksi, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Box Counter Jumlah Repetisi
    cv2.rectangle(frame, (10, 70), (200, 120), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, f"REPS: {counter}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Tampilkan ke layar window
    cv2.imshow('AI Personal Trainer', frame)
    
    # Tekan 'q' untuk keluar dari program utama
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"🏁 Selesai! Sesi latihanmu hari ini totalnya: {counter} gerakan.")