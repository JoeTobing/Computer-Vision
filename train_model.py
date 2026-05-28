import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("🔄 Mencoba membaca dataset...")
try:
    # 1. Load Data
    df = pd.read_csv('dataset_latihan.csv')
    
    # PROTEKSI: Bersihkan data jika ada frame yang gagal deteksi tubuh (NaN)
    df = df.dropna()
    
    print(f"📊 Dataset berhasil dimuat! Total ada {df.shape[0]} baris data siap pakai.")
    print(f"🔹 Jumlah data Push-up (Label 0): {len(df[df['label']==0])}")
    print(f"🔹 Jumlah data Squat (Label 1): {len(df[df['label']==1])}")
    
    # Memisahkan Fitur (X) dan Target/Label (y)
    X = df.drop(columns=['label'])
    y = df['label']
    
    # 2. Split Data (80% Training, 20% Testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\n🧠 Memulai training model dengan Random Forest...")
    # 3. Inisialisasi dan Train Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Evaluasi
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"🏆 Training Selesai! Akurasi Model: {acc * 100:.2f}%")
    
    # 5. Simpan Model menjadi file .pkl
    with open('model_trainer.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("💾 Model berhasil disimpan dengan nama 'model_trainer.pkl'!")

except FileNotFoundError:
    print("❌ Error: 'dataset_latihan.csv' belum ada di folder ini.")
except Exception as e:
    print(f"❌ Ada error lain: {e}")