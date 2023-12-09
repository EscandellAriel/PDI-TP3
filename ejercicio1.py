import cv2
import numpy as np
import matplotlib.pyplot as plt


def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=False, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:
        plt.show(block=blocking)


# --- Leer un video ------------------------------------------------
#cap = cv2.VideoCapture('./tirada_1.mp4')
cap = cv2.VideoCapture('./tirada_1.mp4')

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
# n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

lista = []
contador = 0
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.resize(frame, dsize=(int(width/3), int(height/3)))
        cv2.imshow('Frame',frame)
        contador +=1
        if contador == 70:
            lista.append(frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()

frame = np.array(lista[0])


#frame.shape()

#type(frame)

imshow(frame)

def pre_procesamiento(frame):
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

    return img_gris


def detectar_5(imagen):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(imagen, 8, cv2.CV_32S)

    #plt.figure(), plt.imshow(labels, cmap='inferno'), plt.show(block=False)




    imgContour = imagen

    # Especifica el umbral de área
    area_threshold = (50, 500)  # UMBRAL DE AREA

    # Detectar componentes conectados con stats en la imagen erosionada
    # Filtra las componentes conectadas basadas en el umbral de área
    filtered_labels = []
    filtered_stats = []
    filtered_centroids = []
    componentes_filtradas = np.zeros_like(imgContour)
    for label in range(1, num_labels):  # comienza desde 1 para excluir el fondo (etiqueta 0)
        area = stats[label, cv2.CC_STAT_AREA]

        if area > area_threshold[0] and area < area_threshold[1]:
            filtered_labels.append(label)
            filtered_stats.append(stats[label])
            filtered_centroids.append(centroids[label])
            componentes_filtradas[labels == label] = 255



    min_aspect_ratio = 0.80
    max_aspect_ratio = 1.20
    componentes_filtradas2 = np.zeros_like(imgContour)
    # Filtrar componentes conectadas por relación de aspecto
    filtered_stats_aspect = [stat for stat in filtered_stats if min_aspect_ratio < stat[2] / stat[3] < max_aspect_ratio]
    posiciones=[]



    filtered_centroids_aspect = [centroids[label] for label, stat in zip(filtered_labels, filtered_stats) if any(np.array_equal(stat, s) for s in filtered_stats_aspect)]
    for x in filtered_centroids_aspect:
        posiciones.append(x.tolist())
    filtered_labels_aspect = []
    # Itera sobre las etiquetas y estadísticas filtradas
    for label, stat in zip(filtered_labels, filtered_stats):
        # Verifica si alguna estadística en filtered_stats_aspect es igual a la estadística actual
        for s in filtered_stats_aspect:
            if np.array_equal(stat, s):
                # Si es verdadero, agrega la etiqueta a la lista
                filtered_labels_aspect.append(label)
                componentes_filtradas2[labels == label] = 255
                break  # Sale del bucle interno una vez que se encuentra una coincidencia}


    plt.imshow(imgContour, cmap='gray')

        # Dibuja los bounding boxes de las componentes conectadas filtradas
    for label, stat in zip(filtered_labels_aspect, filtered_stats_aspect):
        x, y, w, h, area = stat

        rect = plt.Rectangle((x, y), w, h, edgecolor='red', linewidth=2, fill=False)
        plt.gca().add_patch(rect)

    return (filtered_labels_aspect, filtered_stats_aspect, posiciones)

nuevo=detectar_5(pre_procesamiento(frame))
viejo=detectar_5(pre_procesamiento(frame))

print(nuevo[2])

prepro=pre_procesamiento(frame)
imshow(prepro)

def contar_dado(stat, imagen):

    x, y, ancho, alto, area = stat


    _, binaria = cv2.threshold(imagen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binaria, 8, cv2.CV_32S)
    dado_recortado_c = cv2.Canny(binaria, 0, 255, apertureSize=3, L2gradient=True)
    objeto_recortado = dado_recortado_c[y:y+alto, x:x+ancho]
    imshow(objeto_recortado, title=num_labels-1)
    return num_labels - 1


print(nuevo[1])

for i in nuevo[1]:
    contar_dado(i, prepro)



def contar_quietos(nuevo, viejo):

    for i, j in zip(nuevo[2], viejo[2]):
        #cont=0 CONTAMOS LOS % JUNTOS O DE A UNO?
        if i[0] == j[0] and i[1] == j[1]:
            return contar_dado(nuevo[1], imagen)
        else:
            print("noigual")



#stats2=contar_quietos(nuevo, viejo)
detectar_5(pre_procesamiento(frame))
cv2.waitKey()
cv2.destroyAllWindows()