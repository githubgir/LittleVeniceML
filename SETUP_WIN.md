## WinPython

## CUDA Toolkit
Install
https://developer.nvidia.com/cuda-downloads

## cudnn
Download and copy the files into C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\
* https://developer.nvidia.com/rdp/cudnn-download

## Numpy
if import scipy complains about MKL (math kernel library) install numpy from wheel as explained below
* http://stackoverflow.com/questions/37267399/importerror-cannot-import-name-numpy-mkl
* http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy

## OpenBLAS
https://github.com/Lasagne/Lasagne/wiki/From-Zero-to-Lasagne-on-Windows-7-(64-bit)#openblas-precompiled

## Tensorflow
https://www.tensorflow.org/install/install_windows
pip3 install --upgrade tensorflow-gpu

## Keras

## Theano
pip3 install --upgrade theano lasagne 

### Theano Configuration
http://deeplearning.net/software/theano/library/config.html
setup file 
```
echo %HOME%
notepad %HOME%\.theanorc
notepad %HOME%\.theanorc.txt
```
### Validate
python -c 'import theano; print(theano.config)' | less



# OLD README
Follow instructions
1) CUDA + cuDNN + VS2013 Community Edition
  * http://linkevin.me/setting-up-cuda-in-windows-10-and-8-7/
  * https://developer.nvidia.com/rdp/cudnn-download
  a) download cuda_check.c and compile it with
  ```
	nvcc -o cuda_check.exe cuda_check.c --compiler-bindir="C:\Program Files (x86)\Microsoft Visual Studio 13.0\VC\bin\amd64" --cl-version=2012
```
	C:\ProgramData\NVIDIA Corporation\CUDA Samples\v7.5

2) WinPython + Theano + Lasagne

https://github.com/Lasagne/Lasagne/wiki/From-Zero-to-Lasagne-on-Windows-7-(64-bit)

3) Install and test the rest, follow

http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/#id1

  a) numpy, pandas, matplotlib, and scikit-learn (should all already be there)
	
  b) theano + lasagne + nolearn
  ```
pip install -u lasagne

pip install --upgrade --no-deps https://github.com/Theano/Theano/archive/master.zip
pip install --upgrade --no-deps https://github.com/Lasagne/Lasagne/archive/master.zip
pip install --upgrade --no-deps https://github.com/dnouri/nolearn/archive/master.zip
```
3c) configure theano
place .theanorc into WinPython\settings

4) https://www.kaggle.com/c/facial-keypoints-detection
