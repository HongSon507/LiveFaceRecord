import numpy as np
from deepface import DeepFace

class FaceEngine:
    def __init__(self):
        self.model_name = "Facenet512" # Model giống code bạn gửi (rất chính xác)
        self.detector_backend = "opencv" # Nhanh nhất cho realtime
        print("--- Đang khởi tạo DeepFace AI... ---")
        
        # Warm-up model
        try:
            dummy = np.zeros((224, 224, 3), dtype=np.uint8)
            DeepFace.represent(dummy, model_name=self.model_name, detector_backend=self.detector_backend, enforce_detection=False)
            print("--- AI đã sẵn sàng! ---")
        except:
            pass

    def get_embedding(self, img_bgr):
        try:
            results = DeepFace.represent(
                img_path = img_bgr, 
                model_name = self.model_name, 
                detector_backend = self.detector_backend,
                align = True, 
                enforce_detection = False 
            )
            
            boxes = []
            embeddings = []
            
            for res in results:
                if 'embedding' in res and 'facial_area' in res:
                    embeddings.append(np.array(res['embedding']))
                    area = res['facial_area']
                    x, y, w, h = area['x'], area['y'], area['w'], area['h']
                    boxes.append([x, y, x+w, y+h])
                    
            return boxes, embeddings
        except:
            return None, None