import numpy as np
import threading
# tinh toan do sai lech giua 2 vector embedding
class FaceMatcher:
    def __init__(self, threshold=0.4): 
        self.threshold = threshold
        self.database = {} 
        self.lock = threading.Lock()

    def register_user(self, name, embedding):
        with self.lock:
            if name not in self.database:
                self.database[name] = []
            self.database[name].append(embedding)

    def search_identity(self, probe_embedding):
        if probe_embedding is None:
            return "Unknown", float('inf')

        min_dist = float('inf')
        identity = "Unknown"

        with self.lock:
            if len(self.database) == 0:
                return "Initializing...", 0.0

            for name, vectors_list in self.database.items():
                for ref_vector in vectors_list:
                    # Công thức cosine dist = 1 - cosine_similarity
                    dot_product = np.dot(probe_embedding, ref_vector)
                    norm_a = np.linalg.norm(probe_embedding)
                    norm_b = np.linalg.norm(ref_vector)
                    cosine_dist = 1 - (dot_product / (norm_a * norm_b))
                    
                    if cosine_dist < min_dist:
                        min_dist = cosine_dist
                        if cosine_dist < self.threshold:
                            identity = name
        
        return identity, min_dist