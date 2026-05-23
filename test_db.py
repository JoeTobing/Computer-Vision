import mysql.connector

try : 
    db = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "",
    database = "personal_trainer"
)
    
    if db.is_connected():
        print ("Berhasil konek")
        
        cursor = db.cursor()
        
        cursor.execute("SHOW TABLES")
        for x in cursor:
            print("Tabel yang ditemukan : ", x)
            
except mysql.connector.Error as err:
    print("Koneksi gagal karena : {err}")

finally :
    if 'db' in locals() and db.is_connected():
        db.close()
        print("Koneksi ditutup aman")    