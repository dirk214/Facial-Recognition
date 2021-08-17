import cv2
import numpy as np
import face_recognition

elon1 = face_recognition.load_image_file('ImagesTrain/Elon-1.jpg')
elon1 = cv2.cvtColor(elon1, cv2.COLOR_BGR2RGB)
elon2 = face_recognition.load_image_file('ImagesTrain/Elon-2.jpg')
elon2 = cv2.cvtColor(elon2, cv2.COLOR_BGR2RGB)

faceLocation = face_recognition.face_locations(elon1)[0]
encodeElon = face_recognition.face_encodings(elon1)[0]
cv2.rectangle(elon1, (faceLocation[3], faceLocation[0]), (faceLocation[1], faceLocation[2]), (255, 0, 255), 2)

faceLocation2 = face_recognition.face_locations(elon2)[0]
encodeElon2 = face_recognition.face_encodings(elon2)[0]
cv2.rectangle(elon2, (faceLocation[3], faceLocation[0]), (faceLocation[1], faceLocation[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeElon], encodeElon2)
faceDis = face_recognition.face_distance([encodeElon], encodeElon2)
print(results, faceDis)
cv2.putText(elon2, f'{results} {round(faceDis[0],2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('Elon Test', elon1)
cv2.imshow('Elon', elon2)
cv2.waitKey(0)