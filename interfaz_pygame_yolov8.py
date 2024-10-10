import pygame
import cv2
import time
import sys
import pygame_widgets
from pygame_widgets.slider import Slider
from pygame_widgets.textbox import TextBox
import numpy as np
from PIL import Image, ImageTk #pip install Pillow
from ultralytics.utils.plotting import Annotator
from ultralytics import YOLO

pygame.init()

# Configuración de la ventana
window_width = 1280
window_height = 680
ventana = pygame.display.set_mode((window_width, window_height))

gris = (38, 38, 38)
azul = (42, 242, 242)

logo = pygame.image.load("logo2.png")
inicio = pygame.image.load("inicio.png")
pausa = pygame.image.load("pausa.png")
captura = pygame.image.load("captura.png")
python = pygame.image.load("python.png")
raspberry = pygame.image.load("raspberry.png")
micro = pygame.image.load("micro.png")
grabar = pygame.image.load("grabar.png")
identificar = pygame.image.load("identificar.png")
ajustar = pygame.image.load("ajustar.png")
brillo = pygame.image.load("brillo.png")
contraste = pygame.image.load("contraste.png")
matiz = pygame.image.load("matiz.png")
automatico = pygame.image.load("automatico.png")

#Creacion de sliders
slider = Slider(ventana, 1050, 440, 150, 15, min=0, max=99, step=1)
output = TextBox(ventana, 1115, 460, 25, 25, fontSize=20)
brightness=slider.getValue()

slider2 = Slider(ventana, 1050, 520, 150, 15, min=0, max=99, step=1)
output2 = TextBox(ventana, 1115, 540, 25, 25, fontSize=20)
contrast=slider2.getValue()

slider3 = Slider(ventana, 1050, 600, 150, 15, min=0, max=99, step=1)
output3 = TextBox(ventana, 1115, 620, 25, 25, fontSize=20)
hue=slider3.getValue()

#Figuras
pygame.draw.rect(ventana, gris, (300, 190, 680, 430))
pygame.draw.rect(ventana, azul, (0, 0, 1280, 80))

### Inicialización de la cámara
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
#cap.set(cv2.CAP_PROP_SETTINGS,1)  Esto permita ajustar brillo y contraste
if not cap.isOpened():
    raise ValueError("No se encontró ninguna cámara web.")

# Redimensionar la cámara
desired_width = 680
desired_height = 430

##opencv dnn para subir la red
##net = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
##model = cv2.dnn_DetectionModel(net)
##model.setInputParams(size=(320,320), scale=1/255)
model = YOLO("BlastocystisSp.pt")
#cargar las clases de la db  de coco de yolo
classesFile = "classes.txt";
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
    COLORS = np.random.uniform(0, 255, size=(len(classes),3))
##classes = []
##with open('classes.txt', 'r') as file_object:
##    for class_name in file_object.readlines():
##        class_name = class_name.strip()
##        classes.append(class_name)

class Button():
    def __init__(self, x, y, image, scale):
        width = image.get_width()
        height = image.get_height()
        self.image = pygame.transform.scale(image, (int(width * scale), int(height * scale)))
        self.rect = self.image.get_rect()
        self.rect.topleft = (x, y)
        self.clicked = False

    def draw(self, surface):
        action = False
        pos = pygame.mouse.get_pos()
        if self.rect.collidepoint(pos):
            if pygame.mouse.get_pressed()[0] == 1 and not self.clicked:
                self.clicked = True
                action = True
        if pygame.mouse.get_pressed()[0] == 0:
            self.clicked = False

        surface.blit(self.image, (self.rect.x, self.rect.y))
        return action

#Botones
b_inicio = Button(100, 200, inicio, 0.9)
b_pausa = Button(100, 300, pausa, 0.9)
b_captura = Button(100, 400, captura, 0.9)
b_grabar = Button(100, 510,grabar,0.9)
b_identificar = Button(1050, 200,identificar,0.9)
# Creación del botón "Automático"
b_automatico = Button(1050, 600, automatico, 0.9)


output.disable() ##Esto es para los valores de los sliders
output2.disable()
output3.disable()

def adjust_brightness_contrast_hue(image, brightness, contrast, hue):
    # Ajustar el brillo y el contraste de la imagen utilizando la fórmula adecuada
    brightness = int((brightness - 50) * 2.55)
    contrast = int((contrast - 50) * 2.55)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[..., 0] = (hsv[..., 0] + hue) % 180

    # Aplicar los ajustes de brillo y contraste a la imagen HSV
    hsv[..., 2] = np.clip(cv2.addWeighted(hsv[..., 2], 1 + contrast / 127.0, hsv[..., 2], 0, brightness - contrast), 0, 255)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    #return np.clip(cv2.addWeighted(image, 1 + contrast / 127.0, image, 0, brightness - contrast), 0, 255)

#Valores automaticos para brillo contraste y matiz
def set_auto_values():
    slider.setValue(45)
    slider2.setValue(70)
    slider3.setValue(25)

run = True
is_camera_running = False
is_recording = False
video_writer = None
session_number=1
while run:
    ventana.blit(logo,(1160,5))
    ventana.blit (python, (1070,5))
    ventana.blit (raspberry,(988,5))
    ventana.blit (micro, (490,275))
    ventana.blit (brillo, (1050,400))
    ventana.blit (contraste, (1050,485))
    ventana.blit (matiz, (1050,570))
    brightness=slider.getValue()
    contrast=slider2.getValue()
    hue=slider3.getValue()
  
    events = pygame.event.get()
    
    for event in events:
        if event.type == pygame.QUIT:
            cap.release()
            pygame.quit()
            exit()

    output.setText(slider.getValue())
    output2.setText(slider2.getValue())
    output3.setText(slider3.getValue())

    if b_automatico.draw(ventana):
        set_auto_values()
                   
    if b_inicio.draw(ventana):
       if not is_camera_running:
            is_camera_running = True
            print("Inicio de la cámara")
        
    if b_pausa.draw(ventana):
        if is_camera_running:
            is_camera_running = False
            identificar= False
            print("Cierre de la cámara")

    if b_captura.draw(ventana):
        if is_camera_running == True:
            print("Captura de imagen")
            #frame_np = np.array(frame_rgb)
##            frame_np = np.rot90(frame_rgb)
            frame_np = pygame.surfarray.array3d(frame_rgb)
                
                # Generar nombre de la imagen
            filename = f"sesion_{session_number}.jpg"

                # Guardar la imagen capturada
            cv2.imwrite(filename, frame_np)
            print("Imagen guardada como:", filename)

                # Incrementar el número de sesión
            session_number += 1
               
            print("Imagen guardada como:", filename)
        else:
            print("La camaradebe estar encendida para realizar una captura")
    if b_grabar.draw(ventana):

        if is_camera_running:
            if not is_recording:
                # Generar nombre de video
                filename = f"sesion_{session_number}.avi"

                # Inicializar VideoWriter
                video_writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"MJPG"), 20, (desired_width, desired_height))
                is_recording = True
                print("Inicio de la grabación")

            else:
                video_writer.release()
                is_recording = False
                print("Video guardado como:", filename)

            # Incrementar el número de sesión
            session_number += 1
        

    if is_camera_running == True:
        # Obtener el cuadro actual de la cámara
        ret, frame = cap.read()

        if ret:
            # Redimensionar el cuadro de la cámara
            frame = cv2.resize(frame, (desired_width, desired_height))
    
            if identificar==True:
                results = model.predict(frame, stream=True,verbose=False)
                for result in results:
                    boxes = result.boxes
                    annotator = Annotator(frame)
            
                    for box in boxes:
                        try:
                            r = box.xyxy[0]                           
                            c = box.cls
                            annotator.box_label(r, label=classes[int(c)], color=COLORS[int(c)])
                        except:
                             pass
            adjusted_frame = adjust_brightness_contrast_hue(frame, brightness, contrast,hue)

            # Convertir el cuadro a formato RGB para Pygame
            frame_rgb = cv2.cvtColor(adjusted_frame, cv2.COLOR_BGR2RGB)
            #dimensiones correctas
            frame_rgb = np.rot90(frame_rgb)
            frame_rgb = pygame.surfarray.make_surface(frame_rgb)
            frame_rgb = pygame.transform.flip(frame_rgb, True, False)  # Voltear imagen horizontalmente
           # window.blit(img, (0, 0))

           # pygame.display.flip()
           # pygame.time.delay(10)

            # Añadir dimensión de lote
           # frame_rgb = np.expand_dims(frame_rgb, axis=0)
            #frame_rgb = pygame.surfarray.make_surface(cv2.cvtColor(adjusted_frame, cv2.COLOR_BGR2RGB))

             
##                (class_ids, scores, bboxes) = model.predict(frame_rgb)
##
##                #dibujar el cuadro
##                for class_id, score, bbox in zip(class_ids, scores, bboxes):
##                        (x, y, w, h)= bbox
##                        class_name = classes[class_id]
##
##                        if class_name == 'person':
##                            cv2.putText(frame_rgb, str(class_name), (x, y-10), cv2.FONT_HERSHEY_PLAIN, 2, (200, 0, 50), 2)
##                            cv2.rectangle(frame_rgb, (x,y), (x+w, y+h), (200, 0, 50), 3)
##                            #print("se esta analizando la clase")
                
##                break
            
            # Crear la imagen de Pygame a partir del cuadro RGB
            #img = pygame.image.frombuffer(frame_rgb.flatten(), (desired_width, desired_height), 'RGB')
            
        # Redimensionar la imagen
            img= pygame.transform.scale(frame_rgb, (desired_width, desired_height))

            # Mostrar la imagen en la ventana
            display_x = 300
            display_y = 190
            ventana.blit(img, (display_x, display_y))

            if b_identificar.draw(ventana):
                identificar = True

                              
    pygame_widgets.update(events)

    pygame.display.update()

pygame.quit()
cv2.destroyAllWindows()


