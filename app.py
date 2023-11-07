from flask import Flask
from flask import render_template
from flask import Response
import cv2
import mediapipe as mp


mp_holistic  = mp.solutions.holistic #holistic model
mp_drawing = mp.solutions.drawing_utils #utilidades de dibujo

def mediapipe_detection(image,model):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) # convierte de BGR a RGB
    image.flags.writeable = False  #La imagen no es writeable
    results = model.process(image)  #image es el frame del open cv, hace la prediccion
    image.flags.writeable = True     #la imagen es writable 
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR) #convierte de RGB a BGR
    return image,results 

def draw_landmarks(image,results): #dibujar las marcas en la imagen
    mp_drawing.draw_landmarks(image,results.face_landmarks,mp_holistic.FACEMESH_TESSELATION) #dibuja las conecciones de la cara
    mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS) #Dibuja las pose conecciones
    mp_drawing.draw_landmarks(image,results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS) #dibuja las conecciones de la mano izquierda
    mp_drawing.draw_landmarks(image,results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS) #dibuja las conecciones de la mano derecha 


def draw_styled_landmarks(image,results): #dibuja los landmarks refinados 
    mp_drawing.draw_landmarks(image,results.face_landmarks,mp_holistic.FACEMESH_TESSELATION,mp_drawing.DrawingSpec(color=(50,110,10),thickness=1,circle_radius=1),mp_drawing.DrawingSpec(color=(80,256,121),thickness=1,circle_radius=1)) #dibuja las conecciones de la cara
    mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS,mp_drawing.DrawingSpec(color=(80,22,10),thickness=2,circle_radius=4),mp_drawing.DrawingSpec(color=(80,44,121),thickness=2,circle_radius=2)) #Dibuja las pose conecciones
    mp_drawing.draw_landmarks(image,results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS,mp_drawing.DrawingSpec(color=(80,22,76),thickness=2,circle_radius=4),mp_drawing.DrawingSpec(color=(121,44,250),thickness=2,circle_radius=2)) #dibuja las conecciones de la mano izquierda
    mp_drawing.draw_landmarks(image,results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS,mp_drawing.DrawingSpec(color=(254,117,66),thickness=2,circle_radius=4),mp_drawing.DrawingSpec(color=(245,66,230),thickness=2,circle_radius=2)) #dibuja las conecciones de la mano derecha 



app = Flask(__name__)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#Detección de caras en formato de cascada
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades +
     "haarcascade_frontalface_default.xml")
def generate():
    with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if ret:
                #hace la deteccion
                image, results = mediapipe_detection(frame,holistic)
                #print(results)
                #dibuja  los landmarks
                draw_styled_landmarks(image,results)
                #La imagen obtenida se convierte en una imagen jpg
                (flag, encodedImage) = cv2.imencode(".jpg", image)
                # En caso de no poder renderizar el frame pasa al siguiente
                if not flag:
                     continue
                
                #yield permite binarizar la imagen jpg en una matriz
                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                        bytearray(encodedImage) + b'\r\n')
            else:
                 break
            
@app.route("/")
def index():
     return render_template("index.html")

@app.route("/video_feed")
def video_feed():
     #Retorna la función de cv2
     return Response(generate(),
          mimetype = "multipart/x-mixed-replace; boundary=frame") #Se ejecuta la imagen y se va a reemplazar frame por frame en el navegador

if __name__ == "__main__":
     app.run(debug=False)

#Se cierra el opencv
cap.release()