import face_recognition
import os, sys
import cv2
import numpy as np
import math
import time

# --- FUNÇÃO DE CONFIANÇA (COM UMA PEQUENA CORREÇÃO) ---
def face_confidence(face_distance, face_match_threshold=0.6):
    range_val = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range_val * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'

# --- CLASSE PRINCIPAL DE RECONHECIMENTO FACIAL ---
class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    def __init__(self):
        os.makedirs('faces', exist_ok=True)
        self.encode_faces()
        # --- Lógica de rastreamento inteligente para desconhecidos ---
        self.saved_unknown_encodings = [] 
        # --- Conjunto para rastrear todas as pessoas conhecidas vistas ---
        self.all_recognized_people = set()

    def encode_faces(self):
        print("A carregar rostos conhecidos...")
        self.known_face_encodings = []
        self.known_face_names = []
        
        for image_name in os.listdir('faces'):
            if image_name.startswith('desconhecido_') or not image_name.endswith(('.jpg', '.png', '.jpeg')):
                continue

            face_image = face_recognition.load_image_file(f'faces/{image_name}')
            
            face_encodings = face_recognition.face_encodings(face_image)
            if len(face_encodings) > 0:
                self.known_face_encodings.append(face_encodings[0])
                self.known_face_names.append(os.path.splitext(image_name)[0])
            else:
                print(f"AVISO: Nenhum rosto encontrado em {image_name}. A ignorar ficheiro.")

        print("Rostos conhecidos carregados:", self.known_face_names)
        # Limpa os desconhecidos salvos na sessão para que possam ser reaprendidos se forem nomeados
        self.saved_unknown_encodings = []

    def autosave_unknown_faces(self, clean_frame, current_face_encodings):
        """ Salva rostos desconhecidos se eles forem novos nesta sessão. """
        for i, name in enumerate(self.face_names):
            if name == "Desconhecido":
                unknown_encoding = current_face_encodings[i]
                
                # Compara o rosto desconhecido atual com os que já foram salvos nesta sessão
                if len(self.saved_unknown_encodings) > 0:
                    distances = face_recognition.face_distance(self.saved_unknown_encodings, unknown_encoding)
                    # Se a distância mínima for muito pequena, é a mesma pessoa, então pulamos
                    if np.min(distances) < 0.6:
                        continue

                # Se chegamos aqui, é um desconhecido novo. Vamos salvá-lo.
                top, right, bottom, left = self.face_locations[i]
                top *= 4; right *= 4; bottom *= 4; left *= 4

                face_image_to_save = clean_frame[top:bottom, left:right]
                
                timestamp = int(time.time())
                file_path = f"faces/desconhecido_{timestamp}.jpg"
                
                cv2.imwrite(file_path, face_image_to_save)
                print(f"Novo rosto desconhecido salvo em '{file_path}'. Pressione 'N' para nomeá-lo.")
                
                # Adiciona o encoding à lista de rastreamento para não salvá-lo novamente
                self.saved_unknown_encodings.append(unknown_encoding)

    def naming_interface(self):
        print("\n--- Interface de Nomenclatura Ativada ---")
        
        unknown_files = [f for f in os.listdir('faces') if f.startswith('desconhecido_')]
        
        if not unknown_files:
            print("Nenhum rosto desconhecido para nomear.")
            return

        for filename in unknown_files:
            file_path = os.path.join('faces', filename)
            image = cv2.imread(file_path)
            
            if image is None: continue

            cv2.imshow('Nomear Rosto', image)
            cv2.waitKey(100)

            prompt = (f"\nDigite o nome para '{filename}'\n"
                      f"  - (p) para pular\n"
                      f"  - (d) para deletar\n"
                      f"Nome: ")
            
            name = input(prompt).strip()

            if name.lower() == 'd':
                os.remove(file_path)
                print(f"Ficheiro '{filename}' deletado.")
            elif name.lower() == 'p' or not name:
                print(f"Ficheiro '{filename}' pulado.")
            else:
                new_filename = f"{name}.jpg"
                new_file_path = os.path.join('faces', new_filename)
                os.rename(file_path, new_file_path)
                print(f"Rosto salvo como '{new_filename}'.")
        
        cv2.destroyWindow('Nomear Rosto')
        print("\n--- Nomenclatura concluída. A recarregar rostos... ---")
        self.encode_faces()

    def run_recognition(self):
        droidcam_url = "http://192.168.1.16:4747/video" # <--- SUBSTITUA PELO SEU IP
        video_capture = cv2.VideoCapture(droidcam_url) 
        
        if not video_capture.isOpened():
            print(f"ERRO: Não foi possível conectar à DroidCam em {droidcam_url}")
            sys.exit("Verifique o endereço IP e se o telemóvel e o PC estão na mesma rede.")

        print(f"Conectado com sucesso à DroidCam em {droidcam_url}")

        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Stream da DroidCam interrompido ou erro na captura."); break

            original_frame = frame.copy()

            if self.process_current_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                recognized_people_in_frame = set()

                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Desconhecido"
                    confidence = "N/A"

                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = self.known_face_names[best_match_index]
                            confidence = face_confidence(face_distances[best_match_index])
                            recognized_people_in_frame.add(name)

                    self.face_names.append(f'{name} ({confidence})' if name != "Desconhecido" else "Desconhecido")
                
                self.autosave_unknown_faces(original_frame, self.face_encodings)
            
            self.process_current_frame = not self.process_current_frame

            self.all_recognized_people.update(recognized_people_in_frame)
            
            # --- ALTERAÇÃO PRINCIPAL: Contagem total correta ---
            # Soma os conhecidos únicos com os desconhecidos únicos salvos nesta sessão
            total_count = len(self.all_recognized_people) + len(self.saved_unknown_encodings)
            count_text = f"Total de Pessoas Unicas: {total_count}"
            cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
            
            help_text = "'N': Nomear Salvos  'Q': Sair"
            cv2.putText(frame, help_text, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top *= 4; right *= 4; bottom *= 4; left *= 4
                color = (0, 255, 0) if "Desconhecido" not in name else (0, 0, 255)
                
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            cv2.imshow('Face Recognition', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('n'): self.naming_interface()

        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()

