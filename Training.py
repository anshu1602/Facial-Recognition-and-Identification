import cv2
import face_recognition

imgAns = face_recognition.load_image_file('Training_images/anshu(20scse1010553).jpg')
imgAns = cv2.cvtColor(imgAns, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('Training_images/anmol(20scse1010698).jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgAns)[0]
encodeAns = face_recognition.face_encodings(imgAns)[0]
cv2.rectangle(imgAns, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeAns], encodeTest)
faceDis = face_recognition.face_distance([encodeAns], encodeTest)
print(results, faceDis)
cv2.putText(imgTest, f'{results} {round(faceDis[0], 2)}', (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('anshu(20scse1010553)', imgAns)
cv2.imshow('Ans Test', imgTest)
cv2.waitKey(0)