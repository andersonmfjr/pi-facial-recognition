# Laptop:
# python encode.py -d dataset -e encodings.pickle -m cnn
# Raspberry Pi:
# python encode.py -d dataset -e encodings.pickle -m hog


from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-m", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())


imagePaths = list(paths.list_images(args["dataset"]))


knownEncodings = []
knownNames = []

for (i, imagePath) in enumerate(imagePaths):
	print("[INFO] Processando imagem {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]


	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	boxes = face_recognition.face_locations(rgb,
		model = args["detection_method"])

	encodings = face_recognition.face_encodings(rgb, boxes)

	for encoding in encodings:
		knownEncodings.append(encoding)
		knownNames.append(name)

# DUMP DATA
print("[INFO] Fazendo gravação dos dados...")

data = {"encodings": knownEncodings, "names": knownNames}
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()