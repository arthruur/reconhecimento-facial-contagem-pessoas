import face_recognition
import os, sys
import cv2
import numpy as np 
import math 


def face_confidence(faceDistance, faceMatchThreshold=0.6):
    if faceDistance > faceMatchThreshold:
        return '0.0%'  # Se a distância for maior que o limite, não há confiança.
    else:
        confidence = 1.0 - faceDistance  # Garantir que a confiança não seja negativa
        return str(round(confidence, 2)) + '%'
    
class FaceRecognition: 
    faceLocations = [] 
    faceEncodings = []
    faceNames = []
    knownFaceEncodings = [] 
    knowsFaceNames = [] 
    processCurrentFrame = True 

    def __init__(self):
        self.encode_faces() 
    
    def encode_faces(self):
        for image in os.listdir('faces'): 
            faceImage = face_recognition.load_image_file(f'faces/{image}')
            faceEncoding = face_recognition.face_encodings(faceImage)[0] 

            self.knownFaceEncodings.append(faceEncoding)
            self.knowsFaceNames.append(image)
    

    def run_recognition(self):
        videoPath = 'WIN_20241025_13_27_43_Pro.mp4'
        videoCapture = cv2.VideoCapture(videoPath)

        if not videoCapture.isOpened():
            sys.exit('Video source not found...')

        # Criar diretório para salvar os frames se não existir
        output_dir = 'output_frames'
        os.makedirs(output_dir, exist_ok=True)

        frame_counter = 0  # Inicializa o contador de frames
        saved_frame_count = 0  # Contador de frames salvos

        while True: 
            ret, frame = videoCapture.read()
            if not ret:
                print("End of video or error encountered.")
                break

            frame_counter += 1  # Aumenta o contador de frames

            # Processa o frame a cada 30 frames
            if frame_counter % 30 == 0:
                '''smallFrame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    rgbSmallFrame = smallFrame[: , :, ::-1]
                    cv2.imshow('small frame',smallFrame)
                    cv2.imshow('rbg small frame',rgbSmallFrame)'''
                    


                # Encontrar todos os rostos no frame atual 
                self.faceLocations = face_recognition.api.face_locations(frame)
                print(self.faceLocations)
                self.faceEncodings = face_recognition.face_encodings(frame, self.faceLocations)

                self.faceNames = []
                if self.faceEncodings:  # Verifique se há codificações faciais
                    for faceEncoding in self.faceEncodings: 
                        matches = face_recognition.compare_faces(self.knownFaceEncodings, faceEncoding)
                        name = 'Unknown'
                        confidence = 'Unknown'

                        faceDistances = face_recognition.face_distance(self.knownFaceEncodings, faceEncoding)


                        bestMatchIndex = np.argmin(faceDistances)


                        if matches[bestMatchIndex]:
                            name = self.knowsFaceNames[bestMatchIndex]
                            confidence = round(1.0 - faceDistances[bestMatchIndex], 2)
                            print(confidence)
                        self.faceNames.append(f'{name} ({confidence})')

                # Anotações de exibição e salvamento do frame
                if self.faceNames:  # Se houver rostos reconhecidos
                    for (top, right, bottom, left), name in zip(self.faceLocations, self.faceNames): 

                        # Adiciona retângulo ao frame
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2) 
                        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
                        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

                    # Salvar o frame anotado
                    output_frame_path = os.path.join(output_dir, f'frame_{saved_frame_count}.jpg')
                    cv2.imwrite(output_frame_path, frame)
                    saved_frame_count += 1  # Aumenta o contador de frames salvos

            # Mostrar o frame processado
            cv2.imshow('Face Recognition', frame)

            # Check for 'q' key press to exit
            if cv2.waitKey(1) == ord('q'):
                break 

        # Liberação de recursos
        videoCapture.release()
        cv2.destroyAllWindows()




if __name__ == '__main__':
    fr = FaceRecognition() 
    fr.run_recognition() 
