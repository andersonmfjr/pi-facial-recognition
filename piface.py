# python piface.py -c haarcascade_frontalface_default.xml -e encodings.pickle

from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
import requests
import json


ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
	help = "path to where the face cascade resides")
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
args = vars(ap.parse_args())


print("[INFO] CARREGANDO ENCODINGS E MODULO DE DETECCAO DE FACE...")
data = pickle.loads(open(args["encodings"], "rb").read())
detector = cv2.CascadeClassifier(args["cascade"])

print("[INFO] INICIALIZANDO O VIDEO...")
# vs = VideoStream(src=0).start()
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

fps = FPS().start()

api = False


while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=500)
	
	# For face detection
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# For face recognition
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	
	rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)

	# OpenCV returns '(x, y, w, h)'
	# Reorder to '(top, right, bottom, left)'
	boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

	
	encodings = face_recognition.face_encodings(rgb, boxes)
	names = []

	# loop over the facial embeddings
	for encoding in encodings:
		matches = face_recognition.compare_faces(data["encodings"],
			encoding)
		name = "Desconhecido"

		# check to see if we have found a match
		if True in matches:
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}

			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1

			name = max(counts, key=counts.get)
			
			if name and api:
				# Colocar função de enviar pro BD ou pra API aqui
				ms = int(round(time.time() * 1000))
				
				# url = "http://elifinho-api.herokuapp.com/professores/{}".format(name)
				url = "https://elifinho-api.herokuapp.com/professores/Flávio-Medeiros"
				payload = { "professor": { "lastseen": ms } }
				
				r = requests.put(url, data = json.dumps(payload))
				
				print(r.text)
				
				print('{} foi visto as {}'.format(name, ms))
		
		names.append(name)

	# loop over the recognized faces
	for ((top, right, bottom, left), name) in zip(boxes, names):
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			0.75, (0, 255, 0), 2)


	cv2.imshow("Pi Facial Recognition - CPSoftware", frame)
	
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

	fps.update()


fps.stop()
print("[INFO] Total time: {:.2f}".format(fps.elapsed()))
print("[INFO] FPS: {:.2f}".format(fps.fps()))


cv2.destroyAllWindows()
vs.stop()
