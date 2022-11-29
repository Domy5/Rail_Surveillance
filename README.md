<!---
https://www.aluracursos.com/blog/como-escribir-un-readme-increible-en-tu-github
https://ajaxhispano.com/ask/comentarios-en-markdown-13816/
https://github-emoji-picker.vercel.app/
https://www.img2go.com/es/convertir-video-a-gif
-->

<h1 align="center"> RAIL SURVEILLANCE </h1>

![Rail_Surveillance](/assets/images/Rail_Surveillance_1.jpg)
![Rail_Surveillance](/assets/images/Rail_Surveillance_2.jpg)
<!---( ![Rail_Surveillance](/assets/images/CC.png) ) -->
   <p align="left">
   <img src="https://img.shields.io/badge/STATUS-In%20Development-red">
   <img src="https://img.shields.io/badge/LICENCE-CC%20(by--nc--nd)-green">
   </p>
   
<h1 align="center"> :octocat: </h1>



Implementation of a RAILWAY SURVEILLANCE software tool that allows the detention of people and/or objects in an area delimited as dangerous on metropolitan train tracks, through existing CCTV cameras.

## :notebook: Index

<!---- [:notebook: Index(#notebook-index)-->
- [:notebook: Index](#notebook-index)
- [:hammer: Project Functionality](#hammer-project-functionality)
- [:hammer_and_wrench: Preparing the environment](#hammer_and_wrench-preparing-the-environment)
- [:page_with_curl::arrow_forward: Program implementation](#page_with_curlarrow_forward-program-implementation)
- [:pushpin: CUDA Toolkit Installation for Nvidea Graphics Cards](#pushpin-cuda-toolkit-installation-for-nvidea-graphics-cards)

## :hammer: Project Functionality 
***
- `Functionality  1:` Alert for the detection of people in the delimited area.
- `Functionality  2:` Alerting for the detection of movements other than trains in the delimited area.
- `Functionality  3:` Emit an audible alarm when detecting hazards in the delimited area.
- `Functionality  4:` Save screenshots of the detected problems.

<h6 align="right">

[:notebook: Index](#notebook-index)
</h6>

## :hammer_and_wrench: Preparing the environment
***
  - **Install Python 3.10 (tested on this version)**
    - https://www.python.org/downloads/release/python-3108/
  
It is possible to install directly in your default environment, but highly recommended to install in a virtual environment as follows:

  - Generate virtual environment with the name ```detection-env```:
    - ```python -m venv deteccion-env```.
  - To activate on Windows installations, run:
    - ```env-detection-scripts-activate.bat```
  - To activate on Unix or MacOS installations, run:
    - ```source detection-env/bin/activate```

Clone the repository in this same folder:
- Clone repository:
  - ```git clone https://github.com/Domy5/Rail_Surveillance.git```


Now there are two possibilities, to have or not to have a GPU, (graphics card compatible with CUDA programming of Nvidea).

- **Installation without GPU:** Install OpenCV with its main modules and extras (contrib).

  - Install OpenCV with its main and extra modules (contrib).
    - ```opencv-contrib-python==4.5.5.62```
    - ```imutils```
  - Install project requirements (numpy, pandas, pyTorch...)
    - ```install -r requirements_without_gpu.txt```

- **Installation with GPU (graphics card compatible with CUDA programming of Nvidea):**
  
  - For installation **OpenCV for GPU**, we need to compile or build our own code from OpenCV source code with CUDA, cuDNN and Nvidia GPU, to do this we need to use some tools like Visual Studio 2016, CMake (GCC C++ compiler), etc.

    - For Windows, run:
    
      - https://thinkinfi.com/install-opencv-gpu-with-cuda-for-windows-10/
      
    - For linux:
    
      - https://towardsdev.com/installing-opencv-4-with-cuda-in-ubuntu-20-04-fde6d6a0a367
      - https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7

  - Install CUDA from Nvidea 11.3 "CUDA Driver Version / Runtime Version          11.5.50"
  - Install CUDA Capability Major/Minor version number:    8.6 (this version will depend on the graphics card you have, in this case NVIDIA GeForce RTX 3060 Ti, 8192MiB, you can check the version in https://en.wikipedia.org/wiki/CUDA)
  - Download the version of cuDNN that matches CUDA in this case "cuDNN v8.0" (save in the corresponding folder usually "C:\Program Files").

    - Have GPU-enabled OpenCV installed (previous step of compiling OpenCV for GPU)
    - pyTorch installation with cuda+cuDNN
      - ```install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 ```
    - Install project requetimientos (pandas, matplotlib...)
      - ```install -r requirements_with_gpu.txt```

We can check what is installed in the environment with this command:

```
python -m torch.utils.collect_env
```
This will result in this console output:

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

<h6 align="right">

[:notebook: Index](#notebook-index)
</h6>

## :page_with_curl::arrow_forward: Program implementation
***

To launch the program it is only necessary to execute the command:

```
python deteccion.py
```

Prealarm example:

![Rail_Surveillance](/assets/images/caida.gif).

Alarm example:

![Rail_Surveillance](/assets/images/caida_1.gif)

Configuration is available through **command line arguments:**

deteccion.py [-h] [-v] [-info] [-m] [-d] [-s] [-mm] [-c {gpu,cpu}] [-i input]

Options:
| Short argument | Long argument | Description |
|:----------|:-------------|:--|
| -h | --help | Show help message and exit|
| -v | --version | Version of the program|
| -info | --information | Information about the versions of the used packages|
| -m | --mask | Show the mask|
| -d | --detection | Display object detections |
| -s | --s-slicer | Show scrollbar (resource-intensive)|
| -mm | --mouse | Displays console click coordinates|
| -c | --process_image | GPU or CPU parameter|
| -i | --input | Video path to process|

Once **in execution** we can modify the behavior of certain program features:

- Options :
  - Esc  :-> Close video execution
  - p    :-> Stop the video
  - c    :-> Captures a frame of the video and saves it in "\\capturas\\numero_img.jpg"
  - s    :-> Activate alarm sound
- OSD :
  - 1:-> Infor Alarm, FPS, frame number...
  - 2:-> ROI
  - 3:-> Outlines inside ROI by background subtraction
  - 4:-> Right foot point
  - 5:-> Rectangle detection, Contours in the scene (People, trains, bags, cars)
  - 6:-> Enable best performance

<h6 align="right">

[:notebook: Index](#notebook-index)
</h6>
  
## :pushpin: CUDA Toolkit Installation for Nvidea Graphics Cards
***

- Have an Nvidea card
- Your graphics card is eligible for the CUDA Toolkit framework https://en.wikipedia.org/wiki/CUDA (example for RTX 3060 Ti is GPU "GA102, GA103, GA104, GA106, GA107" and Micro-architecture Ampere, Compute capability (version) 8.6 )
  - CUDA SDK 11.1 - 11.7 support for compute capability 3.5 - 8.6 (Kepler (in part), Maxwell, Pascal, Volta, Turing, Ampere).
- Have just installed only Python 3.10 (not anaconda).
- Install "numpy"
- Download Community edition Visual Studio, in my case I have downloaded Visual Studio 2019
  - Check "Desktop development with C++", and continue with the defaults and click on install
- Download CMake: https://cmake.org/download/
- Download CUDA from the following link. I have downloaded CUDA Toolkit 11.5
https://developer.nvidia.com/cuda-toolkit-archive
- Download cuDNN according to CUDA
To download cuDNN, you need to register on NVIDIA website, then you can download cuDNNN: https://developer.nvidia.com/rdp/cudnn-archive
  - Browse to the CUDA installation folder, in my case: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.5
      - Copy all files (one file in my case) from the CuDNN bin folder and paste them into the CUDA bin folder (installation folder).
      - Copy all files (one file in my case) from the CuDNN include folder and paste them into the CUDA include folder (installation folder).
      - Copy all files (one file in my case) from the CuDNN lib/x64 folder and paste them inside the CUDA lib/x64 folder (installation folder)
  
By doing so, the cuDNN installation is finished.

- New folder e.g. "Build_CUDA".

- Download https://github.com/opencv/opencv/archive/4.6.0.zip
- Download https://github.com/opencv/opencv_contrib/archive/refs/tags/4.6.0.zip
- Extract the Zip files into the new folder.
- We should change the code "opencv-XX.0cmakeOpenCVDetectPython.cmake" to detect python3 compiler by default.

Previous code :

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
Replace with this code :
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

- Open the Cmake application 
- Provide the path to the OpenCV source code
- Provide the path to the 'Build_CUDA' folder
- Click on the "Configure" button
- In the Configure window, select the optional platform as x64
- Click the "Finish" button
- Now configure the following variables by searching and verifying those variables in the search tab:
  


```
WITH_CUDA — Check it
OPENCV_DNN_CUDA — Check it
ENABLE_FAST_MATH — Check it
OPENCV_EXTRA_MODULES_PATH — Provide path of “modules” directory from “opencv-contrib-X.X.0” C:/Build_CUDA/opencv_contrib-4.6.0/modules"
```
- Press the configuration button again, wait for the "configuration complete" output.
- Now we need to configure some more variables
  
```
CUDA_FAST_MATH — Check it
CUDA_ARCH_BIN — 8.6 (This depends on the graphics card, look in wikipedia the Compute capability in my case for the RTX 3060 Ti is (version) 8.6)
```
- Press the Setup button again, wait for the output of ``Setup completed``.
- After that, click the Generate button and wait for the output " Generating done ".

- With Configuring done and Generating done
Your configuration and code generation are ready . Now you can close the cmake-gui application.

- <span style="color: red;">I have generated a ZIP file in this step.</span>

- Open OpenCV.sln
- Change the "Debug" mode to "Release" mode ("Debug" for "Release")(Located top right options bar)
- Expand " CMakeTargets " (Located on the right), Right click on " ALL_BUILD " and click on build, This may take about 30 minutes to complete.(14:13.15:28)
- Now right click on "INSTALL" (from the same "CMakeTargets") and click compile. This will not take long.
- Verify

```
import cv2
cv2.__version__
cv2.cuda.getCudaEnabledDeviceCount()
```

https://thinkinfi.com/install-opencv-gpu-with-cuda-for-windows-10/

https://thinkinfi.com/use-opencv-with-gpu-python/


![Rail_Surveillance](/assets/images/CC_pequeño.png)

<h6 align="right">

[:notebook: Index](#notebook-index)
</h6>
