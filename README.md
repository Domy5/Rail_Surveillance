# DETECCIÓN DE OBJETOS EN PLATAFORMA DE VÍA

## Preparar el entorno:

  - **Tener instalado Python (probado con la versión 3.10)**

    - Es conveniene usar entornos virtuales como venv:
      - ```python -m venv deteccion-env```
    - Para Windows, ejecuta:
      - ```deteccion-env\Scripts\activate.bat```
    - Para Unix o MacOS, ejecuta:
      - ```source deteccion-env/bin/activate```
    - Clonar repositorio:
      - ```git clone https://github.com/Domy5/Rail_Surveillance.git```

- **Instalación sin GPU:**

  - Instalar OpenCV con sus módulos principales y extras (contrib).
    - ```pip install opencv-contrib-python==4.5.5.62```
    - ```pip install imutils```
  - Instalar los requerimientos del proyecto (numpy, pandas, pyTorch...)
    - ```pip install -r requirements_sin_gpu.txt```

- **Instalación con GPU (Nvidia):**
  
  - Para instalar **OpenCV para GPU**, debemos compilar o construir nuestro propio código del código fuente de OpenCV con CUDA, cuDNN y GPU Nvidia, para hacer esto necesitamos usar algunas herramientas como Visual Studio 2016 (compilador GCC de C++), CMake, etc.

    - Para Windows, ejecuta:
    
      - https://thinkinfi.com/install-opencv-gpu-with-cuda-for-windows-10/
      
    - Para linux:
    
      - https://towardsdev.com/installing-opencv-4-with-cuda-in-ubuntu-20-04-fde6d6a0a367
      - https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7

  - Instalar CUDA de Nvidea  11.3 "CUDA Driver Version / Runtime Version          11.5.50"
  - Instalar CUDA Capability Major/Minor version number:    8.6 (esta versión dependerá de la Tarjeta gráfica que la que se disponga, en este caso NVIDIA GeForce RTX 3060 Ti, 8192MiB, se puede consultar la versión en https://en.wikipedia.org/wiki/CUDA)
  - Descargar la versión de cuDNN que coincida con CUDA en este caso "cuDNN v8.0" (guardar en la carpeta correspondiente normalmente "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.5\bin\")

    - Tener instalado OpenCV habilitado para GPU (paso anterior de compilación OpenCV para GPU)
    - Instalación de pyTorch con cuda+cuDNN
      - ```pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 ```
    - Instalar los requetimientos del proyecto (pandas, matplotlib...)
      - ```pip install -r requirements_con_gpu.txt```

Podemos comprobar lo instalddo en el entorno con este comando:

```
python -m torch.utils.collect_env
```

```
Collecting environment information...
PyTorch version: 1.11.0
Is debug build: False
CUDA used to build PyTorch: 11.3
ROCM used to build PyTorch: N/A

OS: Microsoft Windows 11 Pro
GCC version: Could not collect
Clang version: Could not collect
CMake version: version 3.22.3
Libc version: N/A

Python version: 3.9.7 (default, Sep 16 2021, 16:59:28) [MSC v.1916 64 bit (AMD64)] (64-bit runtime)
Python platform: Windows-10-10.0.22000-SP0
Is CUDA available: True
CUDA runtime version: 11.5.50
GPU models and configuration: GPU 0: NVIDIA GeForce RTX 3060 Ti
Nvidia driver version: 512.96
cuDNN version: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.5\bin\cudnn_ops_train64_8.dll
HIP runtime version: N/A
MIOpen runtime version: N/A

.../
```

## Parámetros:

### Argumentos en línea de comandos:

deteccion.py [-h] [-v] [-info] [-m] [-d] [-s] [-mm] [-c {gpu,cpu}] [-i INPUT]

Opciones:
| Argumento corto | Argumento largo | Descripción |
|:----------|:-------------|:--|
|-h | --help | Mostrar este mensaje de ayuda y salir|
| -v | --version | Versión del programa|
| -info | --informacion | Información de las versiones de los paquetes usados|
| -m | --mascara | Muestra la  mascara|
| -d | --deteccion | Muestra la detecciones de objetos|
| -s | --slicer | Muestra barra de desplazamiento (consume muchos recursos)|
| -mm | --mouse | Muestra por consola las coordenadas de los click|
| -c | --procesar_imagen | Parámetro GPU o CPU|
| -i | --input | Ruta de video a procesar|

### En ejecución:

- Esc : cierra la ejecución del video
- p   : para el video
- c   : captura un frame del video y lo guarda en "C:\\capturas\\numero_img.jpg'"
- o   : OSD alterna la información en pantalla:
  - Sin OSD.
  - Alerta, FPS numero de fotograma.
  - Área de observación.
  - Punto detección persona (inferior derecha).
  - Rectángulo detección (personas, trenes).
