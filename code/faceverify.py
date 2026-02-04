import os
import threading
import cv2
import gc

# cau hinh cho deepface
custom_path = "D:\\face_recog\\models"
os.environ['DEEPFACE_HOME'] = custom_path
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from deepface import DeepFace


REFERENCE_IMG_PATH = "D:\\face_recog\\dataset\\myt.jpg" 


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


counter = 0
is_match = False
thread_is_running = False 
lock = threading.Lock()

def check_face(frame_input):
    global is_match, thread_is_running
    try:
        small_frame = cv2.resize(frame_input, (0, 0), fx=0.5, fy=0.5)
        result = DeepFace.verify(
            img1_path=small_frame, 
            img2_path=REFERENCE_IMG_PATH, 
            model_name='Facenet512', 
            detector_backend='opencv', 
            enforce_detection=False,
            distance_metric='cosine',
            threshold=0.4 
        )
        
        with lock:
            is_match = result['verified']
            
    except Exception as e:
        print(f"Lỗi : {e}")
    finally:
        del frame_input
        gc.collect()
        thread_is_running = False 

while True:
    ret, frame = cap.read()
    if ret:
        # Chỉ chạy nếu chưa có luồng nào đang chạy
        if counter % 30 == 0 and not thread_is_running:
            thread_is_running = True # Khóa lại ngay
            threading.Thread(target=check_face, args=(frame.copy(),), daemon=True).start()
            
            # Dọn rác định kỳ ở luồng chính
            if counter % 300 == 0:
                gc.collect()

        counter += 1

        # Kết quả: hiển thị lên khung hình
        if is_match:
            status_text = "MATCH "
            color = (0, 255, 0)
        else:
            status_text = "NO MATCH !"
            color = (0, 0, 255)

        cv2.putText(frame, status_text, (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        cv2.imshow("Face verify", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()