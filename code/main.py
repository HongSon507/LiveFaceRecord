import cv2
import threading
import os
import gc
import numpy as np
from da_repre import FaceEngine
from khoangcach import FaceMatcher

# --- CẤU HÌNH ---
DATASET_PATH = "D:\\face_recog\\dataset" # Thư mục chứa ảnh người cần nhận diện
custom_path = "D:\\face_recog\\models"
os.environ['DEEPFACE_HOME'] = custom_path
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


current_frame_faces = [] # Lưu kết quả nhận diện của frame hiện tại
thread_is_running = False
lock = threading.Lock()

def recognition_task(frame, engine, matcher):
    global current_frame_faces, thread_is_running
    try:
        boxes, embeddings = engine.get_embedding(frame)
        temp_results = []
        
        if boxes is not None and embeddings is not None:
            for i, box in enumerate(boxes):
                if i < len(embeddings):
                    # 2. Search trong Database
                    name, dist = matcher.search_identity(embeddings[i])
                    temp_results.append((box, name, dist))
        
        # 3. Cập nhật kết quả nhận diện
        with lock:
            current_frame_faces = temp_results
    except Exception as e:
        print(f"Loi: {e}")
    finally:
        thread_is_running = False # Báo hiệu đã xử lý xong
        gc.collect()
      

def load_dataset_background(dataset_path, engine, matcher):
    print(f"--- BẮT ĐẦU QUÉT DATASET TỪ: {dataset_path} ---")
    
    if not os.path.exists(dataset_path):
        print(f"LỖI: Không tìm thấy thư mục {dataset_path}")
        return

    count_person = 0
    
    # Lấy danh sách tất cả file/folder trong dataset
    items = os.listdir(dataset_path)

    for item in items:
        path = os.path.join(dataset_path, item)
        if os.path.isfile(path) and item.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            person_name = os.path.splitext(item)[0]
            print(f"-> Phát hiện ảnh : {item} => Đang học tên: {person_name}")
            
            try:
                # Đọc ảnh (Fix lỗi đường dẫn tiếng Việt)
                stream = open(path, "rb")
                bytes = bytearray(stream.read())
                numpy_array = np.asarray(bytes, dtype=np.uint8)
                img = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
                stream.close()

                if img is not None:
                    _, embs = engine.get_embedding(img)
                    if embs and len(embs) > 0:
                        matcher.register_user(person_name, embs[0])
                        count_person += 1
                        print(f" Đã có: {person_name}")
                    else:
                        print(f" Không thấy mặt trong {item}")
            except Exception as e:
                print(f"   - Lỗi file {item}: {e}")

    print(f"--- DATASET SẴN SÀNG: Đã học {count_person} người ---")

def main():
    global thread_is_running
    
    # 1. Khởi tạo
    engine = FaceEngine()
    matcher = FaceMatcher(threshold=0.4) # Facenet512 + Cosine
    
    # 2. Load dataset ngầm (Để webcam lên hình ngay lập tức)
    threading.Thread(target=load_dataset_background, args=(DATASET_PATH, engine, matcher), daemon=True).start()

    # 3. Main Loop
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    counter = 0
    display_faces = [] # Biến dùng để vẽ lên màn hình
    
    print(" BẤM 'q' ĐỂ THOÁT ---")

    while True:
        ret, frame = cap.read()
        if not ret: break

        if counter % 30 == 0 and not thread_is_running:
            thread_is_running = True
            threading.Thread(target=recognition_task, args=(frame.copy(), engine, matcher), daemon=True).start()

        # Cập nhật biến hiển thị từ kết quả của Thread 
        with lock:
            if current_frame_faces:
                display_faces = current_frame_faces
        
        # --- VẼ HÌNH  ---
        for (box, name, dist) in display_faces:
            x1, y1, x2, y2 = box
            
            if name != "Unknown":
                color = (0, 255, 0) # Xanh lá
                label = f"{name} ({dist:.2f})"
            else:
                color = (0, 0, 255) # Đỏ
                label = f"Unknown ({dist:.2f})"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.54, color, 2)
        cv2.imshow("Face Recognition", frame)
        
        counter += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()