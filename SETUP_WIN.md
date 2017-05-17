## Install Python

## Tensorflow
https://www.tensorflow.org/install/install_windows

## Keras

## Theano
###

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

## CUDA Toolkit
https://developer.nvidia.com/cuda-downloads

## cudnn



# OLD README
Follow instructions
1) CUDA + cuDNN + VS2013 Community Edition
	http://linkevin.me/setting-up-cuda-in-windows-10-and-8-7/
	https://developer.nvidia.com/rdp/cudnn-download
1a) download cuda_check.c and compile it with
	nvcc -o cuda_check.exe cuda_check.c --compiler-bindir="C:\Program Files (x86)\Microsoft Visual Studio 13.0\VC\bin\amd64" --cl-version=2012

	C:\ProgramData\NVIDIA Corporation\CUDA Samples\v7.5

2) WinPython + Theano + Lasagne
https://github.com/Lasagne/Lasagne/wiki/From-Zero-to-Lasagne-on-Windows-7-(64-bit)

3) Install and test the rest, follow
http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/#id1

3a) numpy, pandas, matplotlib, and scikit-learn (should all already be there)
	
3b) theano + lasagne + nolearn
pip install -u lasagne

pip install --upgrade --no-deps https://github.com/Theano/Theano/archive/master.zip
pip install --upgrade --no-deps https://github.com/Lasagne/Lasagne/archive/master.zip
pip install --upgrade --no-deps https://github.com/dnouri/nolearn/archive/master.zip

3c) configure theano
place .theanorc into WinPython\settings

4) https://www.kaggle.com/c/facial-keypoints-detection
