import os
import pickle
import numpy as np
import cv2
import mtcnn
from keras.models import load_model
from . import utils

class TrainDataSet():
    
    def __init__(self, model:str, images_dir:str, encoding_dir:str):
        self.model = model
        self.image = images_dir
        self.encoding = encoding_dir
        
    def start(self) -> bool:
        encodings = dict()
        IMAGE_SIZE = (160, 160)
        DETECTOR = mtcnn.MTCNN()
        ENCODER = load_model(self.model)
        
        for candidate_name in os.listdir(self.image):
            encodes = list()
            candidate_dir = os.path.join(self.image, candidate_name)
            
            for image_name in os.listdir(candidate_dir):
                image_path = os.path.join(candidate_dir, image_name)
                loaded_image = cv2.imread(image_path)
                loaded_image_rgb = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2RGB)
                detect_face = DETECTOR.detect_faces(loaded_image_rgb)
                
                if detect_face:
                    result = max(detect_face, 
                                 key=lambda b: b['box'][2] * b['box'][3])
                    face = utils.normalize(utils.get_face(loaded_image_rgb,
                                            result['box'])[1]
                                            )
                    face = cv2.resize(face, IMAGE_SIZE)
                    
                    encodes.append(ENCODER.predict(
                                    np.expand_dims(face, axis=0))[0]
                                    )
            
            if encodes:
                encode = utils.l2_normalizer.transform(np.expand_dims(
                                                np.sum(encodes, axis=0),
                                                axis=0)
                                                 )[0]
                encodings[candidate_name] = encode
                
        for key in encodings.keys():
            print(f"Trained Dataset for Candidate : {key}")
        try:
            with open(self.encoding, 'bw') as file:
                pickle.dump(encodings, file)
            print ("-----  DATASET TRAINED SUCCESSFULLY  ----- ")
            
            return True
        
        except Exception as e:
            print(e)
            return False