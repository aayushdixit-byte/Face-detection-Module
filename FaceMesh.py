import cv2
import mediapipe as mp
import time
 
class FaceMeshDetector():
    def __init__(self,staticImage=False,maxFaces=2,redefineLms=False,minDetectionCon=0.5,minTrackingCon=0.5):
        self.staticImage = staticImage
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.redefineLms = redefineLms
        self.minTrackingCon = minTrackingCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticImage,self.maxFaces,self.redefineLms,self.minDetectionCon,self.minTrackingCon)
        self.drawSpecs = self.mpDraw.DrawingSpec(thickness=1,circle_radius=1 )

    def findFaceMesh(self,img,draw=True):   
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.result = self.faceMesh.process(imgRGB)
        faces=[]
        if self.result.multi_face_landmarks:
            for faceLms in self.result.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,self.drawSpecs,self.drawSpecs)
                face = []
                for id,lm in enumerate(faceLms.landmark):
                        ih,iw,ic = img.shape
                        x,y = int(lm.x*iw),int(lm.y*ih)
                        face.append([id,x,y])
            faces.append(face)

        return img,faces
        


def main():
    cap  = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceMeshDetector()
    while True:
        success,img = cap.read()
        img = cv2.flip(img, 1)
        img,faces = detector.findFaceMesh(img)
        if len(faces) != 0:
            print(len(faces))
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img,f"FPS:{str(int(fps))}",(10,70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3)
        cv2.imshow("Image",img)
        cv2.waitKey(10)
        
if __name__ == "__main__":
    main()