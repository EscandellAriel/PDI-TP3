import cv2
import numpy as np
import matplotlib.pyplot as plt


##################################################
#FUNCIONES########################################

def redimensionar(frame):
    height, width = frame.shape[:2]
    frame = cv2.resize(frame, dsize=(int(width/3), int(height/3)))
    return frame


def procesar_color(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) # Rangos --> H: 0-179  / S: 0-255  / V: 0-255
    h, s, v = cv2.split(img_hsv)


    # Segmentacion en color - Detectar solo el rojo
    ix_h1 = np.logical_and(h > 180 * .9, h < 180)
    ix_h2 = h < 180 * 0.04
    ix_s = np.logical_and(s > 255 * 0.3, s < 255)
    ix = np.logical_and(np.logical_or(ix_h1, ix_h2), ix_s)
    # ix2 = (ix_h1 | ix_h2) & ix_s   # Otra opcion que da igual...

    r, g, b = cv2.split(img)
    r[ix != True] = 0
    g[ix != True] = 0
    b[ix != True] = 0
    rojo_img = cv2.merge((r, g, b))

    img_gris =  cv2.cvtColor(rojo_img, cv2.COLOR_RGB2GRAY)

    return img_gris,rojo_img


# Función para detectar contornos cuadrados
def detectar_contornos_cuadrados(frame):
    # Convertir a escala de grises
    #gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplicar umbral adaptativo para resaltar contornos
    _, umbral = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Encontrar contornos en la imagen umbralizada
    contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrar contornos cuadrados aproximados
    #contornos_cuadrados = [cnt for cnt in contornos if es_cuadrado(cnt)]
    contornos_cuadrados = [cnt for cnt in contornos if es_cuadrado(cnt) and cv2.contourArea(cnt) > 50*50]
    return contornos_cuadrados

# Función para determinar si un contorno es aproximadamente cuadrado
def es_cuadrado(contorno):
    perimetro = cv2.arcLength(contorno, True)
    approx = cv2.approxPolyDP(contorno, 0.05 * perimetro, True)
    return len(approx) == 4

# Función para recortar regiones de interés de la imagen original
def recortar_contornos(frame, contornos):
    recortes = []
    for contorno in contornos:
        x, y, w, h = cv2.boundingRect(contorno)
        recorte = frame[y:y + h, x:x + w]
        recortes.append(recorte)
    return recortes

def contarDados(recorte):
    _, umbral = cv2.threshold(recorte, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    dado_recortado_para_components = cv2.dilate(umbral, kernel)
        # Aplicar erosión
    imagen_erosionada = cv2.erode(umbral, kernel, iterations=1)
    cv2.imshow('Frame erode', imagen_erosionada)
    # Encuentra componentes conectados
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(imagen_erosionada, 4)

    # Especifica el umbral de área
    area_threshold = (20, 120)  # UMBRAL DE AREA

    # Filtra las componentes conectadas basadas en el umbral de área
    filtered_labels = []
    filtered_stats = []
    filtered_centroids = []

    # Mostrar la imagen con los puntos y bounding boxes
    cv2.imshow('Puntos y Bounding Boxes', recorte)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    for label in range(1, num_labels):  # comienza desde 1 para excluir el fondo (etiqueta 0)
        area = stats[label, cv2.CC_STAT_AREA]

        if area > area_threshold[0] and area < area_threshold[1]:
            x, y, w, h, _ = stats[label]
            relacion_aspecto = float(w) / h
            if relacion_aspecto >=  0.8 and relacion_aspecto <= 1.2:
                filtered_labels.append(label)
                filtered_stats.append(stats[label])
                filtered_centroids.append(centroids[label])
            # Dibujar el bounding box
                cv2.rectangle(recorte, (x, y), (x + w, y + h), (255, 0, 0), 2)
                #cv2.rectangle(umbral, (x, y), (x + w, y + h), (255, 0, 0), 2)
                #plt.imshow(umbral)
                #plt.show()

    return len(filtered_centroids)




# Función para determinar si un dado está quieto de un frame a otro
def dado_quieto(contornos_actual, contornos_anterior):
    # Convertir a escala de grises
    for x,y in zip(contornos_actual,contornos_anterior):
        if np.array_equal(x, y):continue
        else: return False
    return True
    # Calcular la diferencia de área entre frames consecutivos
    diferencia_area = abs(area_dado_actual - area_dado_anterior)

    # Determinar si el dado está quieto (comparando la diferencia de área con el umbral)
    esta_quieto = diferencia_area < umbral_movimiento




#################################################
###Programa######################################

# PROGRAMA
video_path = 'bin/tirada_3.mp4'
cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()
f = 0
contornos_anteriores = None
intervalo_comparacion = 10  # Realizar la comparación cada 5 frames

if not ret:
    print("No se pudo abrir el video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_procesado, frame_color = procesar_color(frame)
    contornos_cuadrados = detectar_contornos_cuadrados(frame_procesado)

    if len(contornos_cuadrados) == 5:
        if f % intervalo_comparacion == 0:  # Realizar la comparación cada 5 frames
            if contornos_anteriores is not None:
                area_actual = sum(cv2.contourArea(contorno) for contorno in contornos_cuadrados)
                area_anterior = sum(cv2.contourArea(contorno) for contorno in contornos_anteriores)

                if abs(area_actual - area_anterior) < 100:  # Ajusta este umbral según tu escenario
                    recortes = recortar_contornos(frame_procesado, contornos_cuadrados)

                    cv2.imshow('Frame Original', redimensionar(frame))
                    cv2.imshow('Frame Original', redimensionar(frame_color))
                    cv2.imshow('Frame Procesado', redimensionar(frame_procesado))

                    for i, recorte in enumerate(recortes):
                        valor = contarDados(recorte)
                        plt.subplot(1, len(recortes), i + 1)
                        plt.imshow(cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB))
                        plt.title(f'Recorte {i + 1}\nPuntos: {valor} Frame {f}')

                    plt.show()

            contornos_anteriores = contornos_cuadrados

    f += 1

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



def grabar_video():
    video_path = './tirada_3.mp4'
    cap = cv2.VideoCapture(video_path)

    # Leer el primer frame para obtener dimensiones
    ret, frame = cap.read()
    if not ret:
        print("No se pudo abrir el video.")
        exit()
    i = 0
    # Bucle para procesar cada cuadro
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i == 70:
        # Aplicar función de procesamiento de color
            frame_procesado, frame_color = procesar_color(frame)

        # Detectar contornos cuadrados
            contornos_cuadrados = detectar_contornos_cuadrados(frame_procesado)
            if len(contornos_cuadrados) == 5:
                contornos_5= contornos_cuadrados
                if contornos_5 != None:
                    print(dado_quieto(contornos_cuadrados,contornos_5))
                pass

        # Recortar regiones de interés de la imagen original
            recortes = recortar_contornos(frame_procesado, contornos_cuadrados)

        # Mostrar el resultado
            cv2.imshow('Frame Original', redimensionar(frame))
            cv2.imshow('Frame Original', redimensionar(frame_color))
            cv2.imshow('Frame Procesado', redimensionar(frame_procesado))

        # Visualizar los recortes utilizando matplotlib (opcional)
            for i, recorte in enumerate(recortes):
                #valor = contar_puntos_en_circulos_alternativo(recorte)
                valor = contarDados(recorte)
                plt.subplot(1, len(recortes), i + 1)
                plt.imshow(cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB))
                plt.title(f'Recorte {i + 1}\nPuntos: {valor}')
                

            plt.show()
        i +=1
        # Salir del bucle si se presiona 'q'
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()


