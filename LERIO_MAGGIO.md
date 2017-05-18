# Lerio Maggio Tutorial
```
git clone https://github.com/leriomaggio/deep-learning-keras-tensorflow.git
```
## Required versions
```
numpy: 1.11.1
scipy: 0.18.0
matplotlib: 1.5.2
iPython: 5.1.0
scikit-learn: 0.18
keras:  2.0.2
Theano:  0.9.0
Tensorflow:  1.0.1
```

## Update instructions
```
pip install --update pip
conda create -n tensorflow
source activate tensorflow

pip install --upgrade \
numpy==1.11.1 \
scipy==0.18.0 \
matplotlib==1.5.2 \
ipython==5.1.0 \
scikit-learn==0.18 \
keras==2.0.2 \
theano==0.9.0 \
lasagne
```
## Update instructions for Tensorflow
* https://www.tensorflow.org/install/install_linux#InstallingAnaconda
* https://www.tensorflow.org/install/install_linux#the_url_of_the_tensorflow_python_package

```
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.1.0-cp27-none-linux_x86_64.whl
#or
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.1.0-cp27-none-linux_x86_64.whl
#or
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.0.1-cp27-none-linux_x86_64.whl
```

## Upgrade NVIDIA drivers, CUDA, etc
* http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/accelerated-computing-instances.html
* https://www.pugetsystems.com/labs/hpc/Install-Ubuntu-16-04-or-14-04-and-CUDA-8-and-7-5-for-NVIDIA-Pascal-GPU-825/
* http://markus.com/install-theano-on-aws/
* http://christopher5106.github.io/big/data/2016/01/27/two-AMI-to-create-the-fastest-cluster-with-gpu-at-the-minimal-engineering-cost-with-EC2-NVIDIA-Spark-and-BIDMach.html


### NVIDIA display driver
```
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install nvidia-367
```

### CUDA Toolkit Download
https://developer.nvidia.com/cuda-downloads
```
sudo wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run
chmod 755 cuda_*
sudo bash cuda_8.0.61_375.26_linux-run
sudo bash ./cuda_7.5.18_linux.run --silent --toolkit --samples --samplespath=/usr/local/cuda-7.5/samples --override
sudo bash ./cuda_8.0.27_linux.run --silent --toolkit --samples --samplespath=/usr/local/cuda-8.0/samples --override
# follow instructions

```
### cudnn
http://deeplearning.net/software/theano/library/sandbox/cuda/dnn.html
http://www.pyimagesearch.com/2016/07/04/how-to-install-cuda-toolkit-and-cudnn-for-deep-learning/
https://developer.nvidia.com/rdp/cudnn-download
download cudnn v5 for cuda v8, otherwise won't work
download requires login use my hotmail.co.uk and simple small letter pass do download file xxx.solitairetheme8
then upload to ec2 via pscp
```
pscp -i C:\Dev\AWS\AWSKeyPair2.ppk C:\Users\Andreas\Downloads\cudnn-8.0-linux-x64-v5.0-ga.solitairetheme8 %AWS_USER%@%AWS_URL%:/home/%AWS_USER%/
```
```
cd ~
tar -zxf cudnn-7.5-linux-x64-v5.0-ga.tgz
cd cuda
sudo cp lib64/* /usr/local/cuda/lib64/
sudo cp include/* /usr/local/cuda/include/
```


## Start Jupyter Notebook in TensorFlow environment
```
source activate tensorflow
jupyter notebook
```
## Configure Keras
```
nano ~/.keras/keras.json
```
with following content
```
{
	"epsilon": 1e-07,
	"backend": "tensorflow",
	"floatx": "float32",
	"image_data_format": "channels_last"
}
```

## Configure Theano
```
THEANO_FLAGS='floatX=float32,device=gpu,mode=FAST_RUN,fastmath=True' jupyter notebook

or

nano ~/.theanorc
```
with following content
```
[global]
floatX=float32
device=gpu
[mode]=FAST_RUN

[nvcc]
fastmath=True

[cuda]
root=/usr/local/cuda
```
