- Tener una tarjeta Nvidea
- su tarjeta gráfica es elegible para el marco CUDA Toolkit https://en.wikipedia.org/wiki/CUDA (ejemplo para RTX 3060 Ti es GPU "GA102, GA103, GA104, GA106, GA107" y Micro-architecture Ampere, Compute capability (version) 8.6 )
  - CUDA SDK 11.1 – 11.7 support for compute capability 3.5 – 8.6 (Kepler (in part), Maxwell, Pascal, Volta, Turing, Ampere).

# Instalación

- Tener recien intalado solo Python 3.10.5 (no anaconda)
- Instalar “numpy” (pip install numpy)
- Descargar Community edition Visual Studio, en mi caso he descargado Visual Studio 2019
  -  Verifique "Desarrollo de escritorio con C ++", y Continúe con los valores predeterminados y haga clic en instalar
- Descargar CMake: https://cmake.org/download/
- Descargar CUDA desde el siguiente enlace. He descargado CUDA Toolkit 11.5
https://developer.nvidia.com/cuda-toolkit-archive
- Descargar cuDNN según CUDA
Para descargar cuDNN, debe registrarse en el sitio web de NVIDIA, luego puede descargar cuDNN: https://developer.nvidia.com/rdp/cudnn-archive
  -   Busque la carpeta de instalación de CUDA, en mi caso: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.5
      -   Copie todos los archivos (un archivo en mi caso) de la carpeta bin de CuDNN y péguelos dentro de la carpeta bin de CUDA (carpeta de instalación)
      -   Copie todos los archivos (un archivo en mi caso) de la carpeta de inclusión de CuDNN y péguelos dentro de la carpeta de inclusión de CUDA (carpeta de instalación )
      -   Copie todos los archivos (un archivo en mi caso) de la carpeta lib/x64 de CuDNN y péguelos dentro de la carpeta lib/x64 de CUDA (carpeta de instalación)
  
Al hacer eso, la instalación de cuDNN ya está finalizada.

- Carpeta Nueva por ejemplo "Build_CUDA"

- Descargar https://github.com/opencv/opencv/archive/4.6.0.zip
- Descargar https://github.com/opencv/opencv_contrib/archive/refs/tags/4.6.0.zip
- Extaer los archivos Zip e n la nueva carpeta.
- Deberíamos cambiar el código “opencv-XX.0\cmake\OpenCVDetectPython.cmake” para que detecte el compilador python3 por defecto.

Código anterior :

```
if(PYTHON_DEFAULT_EXECUTABLE)
    set(PYTHON_DEFAULT_AVAILABLE "TRUE")
elseif(PYTHON2_EXECUTABLE AND PYTHON2INTERP_FOUND)
    # Use Python 2 as default Python interpreter
    set(PYTHON_DEFAULT_AVAILABLE "TRUE")
    set(PYTHON_DEFAULT_EXECUTABLE "${PYTHON2_EXECUTABLE}")
elseif(PYTHON3_EXECUTABLE AND PYTHON3INTERP_FOUND)
    # Use Python 3 as fallback Python interpreter (if there is no Python 2)
    set(PYTHON_DEFAULT_AVAILABLE "TRUE")
    set(PYTHON_DEFAULT_EXECUTABLE "${PYTHON3_EXECUTABLE}")
endif()
```
Reemplace con este código :
```
if(PYTHON_DEFAULT_EXECUTABLE)
    set(PYTHON_DEFAULT_AVAILABLE "TRUE")
elseif(PYTHON3INTERP_FOUND) 
 # Use Python 3 as default Python interpreter
    set(PYTHON_DEFAULT_AVAILABLE "TRUE")
    set(PYTHON_DEFAULT_EXECUTABLE "${PYTHON3_EXECUTABLE}")
elseif(PYTHON2INTERP_FOUND) 
    # Use Python 2 as fallback Python interpreter (if there is no Python 3)
    set(PYTHON_DEFAULT_AVAILABLE "TRUE")
    set(PYTHON_DEFAULT_EXECUTABLE "${PYTHON2_EXECUTABLE}")
endif() 
```

- Abra la aplicación Cmake 
- Proporcione la ruta del código fuente de OpenCV
- Proporcione la ruta de la carpeta 'Build_CUDA'
- Haga clic en el botón " Configurar "
- En la ventana Configurar, seleccione la plataforma opcional como x64
- Haga clic en el botón " Finalizar "
- Ahora configure las siguientes variables buscando y verificando esas variables en la pestaña de búsqueda:
  


```
WITH_CUDA — Check it
OPENCV_DNN_CUDA — Check it
ENABLE_FAST_MATH — Check it
OPENCV_EXTRA_MODULES_PATH — Provide path of “modules” directory from “opencv-contrib-X.X.0” C:/Build_CUDA/opencv_contrib-4.6.0/modules"
```
- Presiona el botón de configuración nuevamente, espera la salida de " configuración finalizada "
- Ahora necesitamos configurar algunas variables más
  
```
CUDA_FAST_MATH — Check it
CUDA_ARCH_BIN — 8.6 (Esto depende de la tarjeta grafica, buscar en wikipedia el Compute capability en mi caso para la RTX 3060 Ti es (version) 8.6)
```
- Presiona el botón de configuración nuevamente, espera la salida de " configuración finalizada "
- Después de eso, haga clic en el botón Generar y espere la salida " Generación finalizada ".

- Con Configuring done y Generating done
Su configuración y generación de código están listas . Ahora puedes cerrar la aplicación cmake-gui

- <span style="color: red;">He generado un ZIP en este paso</span>

- Abrir OpenCV.sln
- Cambie el modo "depurar" al modo "liberar" ("Debug" por "Release")(Ubicado barra de opciones arriba derecha)
- Expanda “ CMakeTargets ” (Ubicado a la derecha), Haga clic derecho en " ALL_BUILD " y haga clic en construir, Esto puede tardar unos 30 minutos en completarse.(14:13.15:28)
- Ahora haga clic con el botón derecho en " INSTALAR " (del mismo "CMakeTargets") y haga clic en compilar . Esto no llevará mucho tiempo.
- Verificar

```
import cv2
cv2.__version__
cv2.cuda.getCudaEnabledDeviceCount()
```

https://thinkinfi.com/install-opencv-gpu-with-cuda-for-windows-10/

https://thinkinfi.com/use-opencv-with-gpu-python/