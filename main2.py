import cv2
import mediapipe as mp
import time
import numpy as np
import csv
import os

class FaceMeshDetector():
    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, refine_landmarks=True):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.refine_landmarks = refine_landmarks

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, self.refine_landmarks, self.minDetectionCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec, self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x, y])
                faces.append(face)
        return img, faces

def save_landmarks(faces, file_path, face_id):
    if not faces:
        return
    with open(file_path, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for face in faces:
            row = [face_id]
            for landmark in face:
                row.extend(landmark)
            csvwriter.writerow(row)

def main():
    cap = cv2.VideoCapture("Video2.mp4")
    pTime = 0
    detector = FaceMeshDetector()
    face_id = "face_1"  # Example face ID, you can use a different method to get the actual face ID
    output_file = "face_landmarks.csv"

    if not os.path.exists(output_file):
        with open(output_file, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            headers = ["face_id"]
            for i in range(468):  # 468 landmarks
                headers.extend([f"x_{i}", f"y_{i}"])
            csvwriter.writerow(headers)

    while True:
        success, img = cap.read()
        if not success:
            break
        img, faces = detector.findFaceMesh(img)
        save_landmarks(faces, output_file, face_id)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
