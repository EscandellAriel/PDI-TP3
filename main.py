import os
from funciones import *

# Obtener la lista de archivos de imágenes en el directorio actual
archivos_video = [f for f in os.listdir('./') if f.endswith(('.mp4'))]


for video in archivos_video:
  programa_dados(video)
  grabar_video(video)