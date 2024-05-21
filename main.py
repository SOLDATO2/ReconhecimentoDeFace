import cv2
import mediapipe as mp
import time
import numpy as np

class FaceMeshDetector():
    def __init__(self, staticMode = False, maxFaces=2, minDetectionCon=0.5, refine_landmarks = True):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.refine_landmarks = refine_landmarks

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, self.refine_landmarks, self.minDetectionCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2) #grossura e radius da mascara
        
    def findFaceMesh(self, img, draw = True):
            self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results = self.faceMesh.process(self.imgRGB)
            faces = []
            if self.results.multi_face_landmarks:
                for faceLms in self.results.multi_face_landmarks:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec, self.drawSpec) #Facemesh_countours são os dados que precisam ser extraidos

                #printando landmarks
                face = []
                for id, landmark_atual in enumerate(faceLms.landmark):
                    #convertendo landmarks para pixels
                    ih, iw, ic = img.shape
                    x,y = int(landmark_atual.x*iw), int(landmark_atual.y*ih)
                    #print("->",id,x,y)
                    face.append([x,y])
                faces.append(face)
            
            return img, faces



def main():
    cap = cv2.VideoCapture("Videos/Video1.mp4")
    pTime = 0
    detector = FaceMeshDetector()
    



    while True: # para cada frame
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)
        #######
        #quantia de faces detectadas
        print(len(faces))
        ##########
        #calcular fps
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}',(20,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
        ###
        cv2.imshow("Image", img) #titulo e imagem capturada
        cv2.waitKey(1)
    


if __name__ == "__main__":
    main()
#parei em 26:57 https://www.youtube.com/watch?v=V9bzew8A1tc


