import cv2
import face_recognition as fr 


# Primeira etapa: Localizar o rosto na imagem 

imgArthur = fr.load_image_file('faces/kauan.jpg')
imgArthur = cv2.cvtColor(imgArthur, cv2.COLOR_BGR2RGB)

imgArthurTest = fr.load_image_file('faces/arthur.jpg')
imgArthurTest = cv2.cvtColor(imgArthurTest, cv2.COLOR_BGR2RGB)


faceLoc = fr.face_locations(imgArthur)[0]
cv2.rectangle(imgArthur, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (0,255,0), 2)
print(faceLoc)

cv2.imshow('Arthur', imgArthur)
cv2.imshow('ArthurTest', imgArthurTest)

# Encoding do rosto

encodeArthur = fr.face_encodings(imgArthur)[0]
encodeArthurTest = fr.face_encodings(imgArthurTest)[0]


comparacao = fr.compare_faces([encodeArthur], encodeArthurTest)

print(comparacao)

cv2.waitKey(0)