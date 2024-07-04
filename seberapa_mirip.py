import face_recognition
import cv2
import numpy as np
from face_recognition.face_recognition_cli import image_files_in_folder
from pathlib import Path
from sklearn.metrics import accuracy_score
import math


# Baca sumber camera/webcam
video_capture = cv2.VideoCapture(0)

face_image = {}
face_encoding = {}
known_face_encodings = []
known_face_names = []

# Membaca image dalam folder "muha"
for img_path in image_files_in_folder("muha"):
    face_image[Path(img_path).stem] = face_recognition.load_image_file(img_path)
    face_encoding[Path(img_path).stem] = face_recognition.face_encodings(face_image[Path(img_path).stem])[0]
    known_face_encodings.append(face_encoding[Path(img_path).stem])
    known_face_names.append(Path(img_path).stem)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# fungsi prosentase jarak encoding
def face_distance_to_conf(face_distance, face_match_threshold=0.6):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))

while True:
    # Ambil 1 frame dari video
    ret, frame = video_capture.read()

    # Mencegah frame terbaca berkali-kali
    if process_this_frame:
        # Mengubah ukuran frame jadi 1/4 untuk mempercepat proses pengenalan wajah
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert gamabar dari BGR color (yang digunakan OpenCV) ke RGB color (dibutukan untuk face_recognition)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
        
        # Cari semua wajah dan face encodings dalam frame ini 
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Melihat apakah wajah cocok
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = ""
            acc = 0


            # Bandingkan jarak terdekat dari encoding file wajah dengan wajah yang baru dibaca
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
				# menghitung akurasi
                acc = int(face_distance_to_conf(face_distances[best_match_index]) * 100)

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Tampilkan hasil
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Besarkan lagi skala gambar menjadi 1/4 kali
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Menggambar kotak di wajah
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Menggabar tulisan dibawah wajah
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        if(acc>0):
            cv2.putText(frame, name + " " + str(acc) + "%", (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)

    # Menampilkan video hasil
    cv2.imshow('Video', frame)

    # Tekan "q" untuk kelura aplikasi
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bebaskan webcam
video_capture.release()
cv2.destroyAllWindows()