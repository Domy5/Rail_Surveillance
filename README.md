# DETECCIÓN DE OBJETOS EN PLATAFORMA DE VÍA

## Trabajo Fin de Grado Domingo Martínez Núñez


Preparando el entorno:

- Instalación sin GPU:

  - Tener Python instalado (probado con la versión 3.10)
  - Clonar repositorio con sus submódulos:
    - ```git clone --recurse-submodules https://github.com/Domy5/Rail_Surveillance.git```
     
  - pip install opencv-contrib-python=4.5.5
  - Instalar ```pip install -r requirements_sin_gpu.txt```

- Instalación con GPU Nvidea:
  
Para instalar OpenCV+GPU en Windows, debemos compilar o construir nuestro porpio código del código fuente de OpenCV con CUDA, cuDNN y GPU Nvidia. 
Para hacer eso necesitamos usar algunas herramientas como Visual Studio 2016 (compilador GCC de C++), CMake, etc.

  - https://thinkinfi.com/install-opencv-gpu-with-cuda-for-windows-10/

Hay que instalar CUDA de Nvidea  11.3 "CUDA Driver Version / Runtime Version          11.5.50"
CUDA Capability Major/Minor version number:    8.6 (esta versión dependerá de la Tarjeta gráfica que la que se disponga, en este caso NVIDIA GeForce RTX 3060 Ti, 8192MiB se puede consultar la versión en https://en.wikipedia.org/wiki/CUDA)
después la versión que coincida de cuDNN "Download cuDNN v8.0"

- Tener instalado Opencv 4.5.5 habilitado para GPU
- Instalación de pyTorch con cuda+cuDNN
 
  - ```pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio===0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html```

Como comprobar el entorno:
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

Versions of relevant libraries:
[pip3] mypy-extensions==0.4.3
[pip3] numpy==1.20.3
[pip3] numpydoc==1.1.0
[pip3] torch==1.11.0
[pip3] torchaudio==0.11.0
[pip3] torchvision==0.12.0
[conda] blas                      1.0                         mkl
[conda] cudatoolkit               11.3.1               h59b6b97_2
[conda] mkl                       2021.4.0           haa95532_640
[conda] mkl-service               2.4.0            py39h2bbff1b_0
[conda] mkl_fft                   1.3.1            py39h277e83a_0
[conda] mkl_random                1.2.2            py39hf11a4ad_0
[conda] mypy_extensions           0.4.3            py39haa95532_0
[conda] numpy                     1.20.3           py39ha4e8547_0    anaconda
[conda] numpy-base                1.20.3           py39hc2deb75_0
[conda] numpydoc                  1.1.0              pyhd3eb1b0_1
[conda] pytorch                   1.11.0          py3.9_cuda11.3_cudnn8_0    pytorch
[conda] pytorch-mutex             1.0                        cuda    pytorch
[conda] torch                     1.11.0                   pypi_0    pypi
[conda] torchaudio                0.11.0               py39_cu113    pytorch
[conda] torchvision               0.12.0                   pypi_0    pypi
```
- Instalar ```pip install -r requirements.txt```


