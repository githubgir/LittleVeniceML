# LittleVeniceML
Little Venice Machine Learning Group

# Linux Shall Commands
* nano text editor
* git clone <repository>
* sudo = super user do
* lsb_release -a

# Setup on AWS
## Launch AWS EC2
http://cs231n.github.io/aws-tutorial/
* zones
  * us-west-1 
  * eu-west-1
* instances
  * m3xlarge
  * g2x2large
* open ports
  * 22 SSH
  * 8888 Jupyter Notebook
  * 6006 TensorBoard

## Putty SSH to AWS from Windows
* http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/putty.html
* http://techexposures.com/how-to-copy-files-to-aws-ec2-server-from-windows-pc-command-prompt/

Authorise SSH access to EC2 - i.e. open port 22
http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/authorizing-access-to-an-instance.html

* PuttyGen to convert downloaded pem file to pkk
* Putty GUI or putty cmd to ssh
* pscp to securely transfer files
```
putty -ssh -i C:\Dev\AWS\AWSKeyPair2.ppk ec2-user@ec2-54-183-144-206.us-west-1.compute.amazonaws.com

pscp -i C:\Dev\AWS\AWSKeyPair2.ppk C:\Dev\AWS\AWSKeyPair2.pem ec2-user@ec2-54-183-144-206.us-west-1.compute.amazonaws.com:/home/ec2-user/AWSKeyPair2.pem
```

## NVIDIA Monitor
```
nvidia-smi -l 1
```

## Jupyter Notebook
http://efavdb.com/deep-learning-with-jupyter-on-aws/
* dont specify password otherwise not working
* dont specify certfile and key file otherwise not working on iPad
* open port 8888
* open browser on http(s)://<Public_DNS_Address>:8888

``` bash
#!/bin/bash

CERTIFICATE_DIR="/home/ubuntu/certificate"
JUPYTER_CONFIG_DIR="/home/ubuntu/.jupyter"
THEANO_CONFIG_DIR="/home/ubuntu"

if [ ! -d "$CERTIFICATE_DIR" ]; then
    mkdir $CERTIFICATE_DIR
    openssl req -x509 -nodes -days 365 -newkey rsa:1024 -keyout "$CERTIFICATE_DIR/mykey.key" -out "$CERTIFICATE_DIR/mycert.pem" -batch
    chown -R ubuntu $CERTIFICATE_DIR
fi

if [ ! -f "$JUPYTER_CONFIG_DIR/jupyter_notebook_config.py" ]; then
    # generate default config file
    #jupyter notebook --generate-config
    mkdir $JUPYTER_CONFIG_DIR

    # append notebook server settings
    cat <<EOF >> "$JUPYTER_CONFIG_DIR/jupyter_notebook_config.py"
# Set options for certfile, ip, password, and toggle off browser auto-opening
#c.NotebookApp.certfile = u'$CERTIFICATE_DIR/mycert.pem'
#c.NotebookApp.keyfile = u'$CERTIFICATE_DIR/mykey.key'
# Set ip to '*' to bind on all interfaces (ips) for the public server
c.NotebookApp.ip = '*'
#c.NotebookApp.password = u''
c.NotebookApp.open_browser = False

# It is a good idea to set a known, fixed port for server access
c.NotebookApp.port = 8888
EOF
    chown -R ubuntu $JUPYTER_CONFIG_DIR
fi
```

## iPad
* SSH on iPad
http://www.packetnerd.com/?p=131
* Jupyter Notebook on iPad 
1) don't specify cerfiles in the script above, or
2) properly via Certificate Authority (own or 3rd party)
  * http://jupyter-notebook.readthedocs.io/en/latest/public_server.html
  * https://letsencrypt.org/getting-started/

## Issues
### Spot instances greyed out for Market AMIs
* https://www.paulwakeford.info/2016/01/07/aws-marketplace-and-spot-instances/
* https://aws.amazon.com/marketplace/pp/B06VSPXKDX

# Setup from Scratch
* http://markus.com/install-theano-on-aws/
* http://christopher5106.github.io/big/data/2016/01/27/two-AMI-to-create-the-fastest-cluster-with-gpu-at-the-minimal-engineering-cost-with-EC2-NVIDIA-Spark-and-BIDMach.html

# Lerio Maggio Tutorial
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

# AWS CLI Command Line Interface
Run on windows to create EC2 with spark-ec2 installed, then ssh on it and start cluster

```
aws configure
# AWS Access Key ID [****************HVHA]:
# AWS Secret Access Key [****************mfU3]:
# Default region name [us-west-1]:
# Default output format [json]:
```

List of AMIs supported by spark e.g. ami-72320f37
https://github.com/amplab/spark-ec2/blob/v4/ami-list/us-west-1/hvm


For some reason I cannot fine the needed AMI via EC2 dashboard, therefore need to launch via AWS CLI
```
aws ec2 describe-images --image-ids ami-72320f37
aws ec2 run-instances --image-id ami-72320f37 --instance-type m4.large --key-name AWSKeyPair2 --security-groups sec-grp-ssh-jupyter-spark
aws ec2 request-spot-instances --spot-price "0.1" --launch-specification file://C:/Dev/AWS/spot.json
aws ec2 describe-instances --query "Reservations[*].Instances[*].[InstanceId, ImageId, PublicDnsName]"
aws ec2 describe-instances --query "Reservations[*].Instances[*].[InstanceId,ImageId,PublicDnsName]" --filters "Name=image-id,Values=ami-72320f37"

set AWS_URL=ec2-54-193-106-52.us-west-1.compute.amazonaws.com
set AWS_USER=ec2-user

putty -ssh -i C:\Dev\AWS\AWSKeyPair2.ppk %AWS_USER%@%AWS_URL%
pscp -i C:\Dev\AWS\AWSKeyPair2.ppk C:\Dev\AWS\AWSKeyPair2.pem %AWS_USER%@%AWS_URL%:/home/%AWS_USER%/AWSKeyPair2.pem

"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" "https:\\%AWS_URL%:8888"
"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" "https:\\%AWS_URL%:6006"
```

This requires spot.json file with spot request configuration
```
{
  "ImageId": "ami-72320f37",
  "KeyName": "AWSKeyPair2",
  "SecurityGroupIds": [ "sg-66dbf801" ],
  "InstanceType": "m4.large",
  "Placement": {
    "AvailabilityZone": "us-west-1b"
  }
}
```


# Spark Clusters
Setup instructions for AMI with Spark and ML libraries
* http://christopher5106.github.io/big/data/2016/01/27/two-AMI-to-create-the-fastest-cluster-with-gpu-at-the-minimal-engineering-cost-with-EC2-NVIDIA-Spark-and-BIDMach.html
* https://github.com/amplab/spark-ec2.git
* https://github.com/christopher5106/spark-ec2
* http://ampcamp.berkeley.edu/3/exercises/launching-a-bdas-cluster-on-ec2.html

```
chmod -c 600 AWSKeyPair2.pem
# can connect to other instances
ssh -i AWSKeyPair2.pem ec2-user@ec2-54-241-142-236.us-west-1.compute.amazonaws.com

git clone https://github.com/amplab/spark-ec2.git

export AWS_ACCESS_KEY_ID=<put your stuff here>
export AWS_SECRET_ACCESS_KEY=<put your stuff here>
./spark-ec2/spark-ec2 \
--key-pair AWSKeyPair2 \
--identity-file ~/AWSKeyPair2.pem \
--region=us-west-1 \
--instance-type=m3.large \
--slaves=1 \
--hadoop-major-version=2 \
--spot-price="0.1" \
--ami=ami-72320f37 \
launch spark-cluster

#other options
--instance-profile-name=spark-ec2 \
--ami=
--spark-ec2-git-repo=https://github.com/christopher5106/spark-ec2 \

./spark-ec2/spark-ec2 -k AWSKeyPair2 -i ~/AWSKeyPair2.pem \
--region=us-west-1 login spark-cluster


# launch the shell
./spark/bin/spark-shell

# terminate the cluster
./spark-ec2/spark-ec2 -k AWSKeyPair2 -i ~/AWSKeyPair2.pem \
--region=us-west-1  destroy spark-cluster
```

## PySpark install and Jupyter notebook configuration
https://www.dataquest.io/blog/pyspark-installation-guide/

# Other ways than AWS EC2
* AWS ML
* AWS AI
https://aws.amazon.com/amazon-ai/
* AWS CloudFormation for Deep Learning
https://aws.amazon.com/blogs/compute/distributed-deep-learning-made-easy/
https://github.com/awslabs/deeplearning-cfn
* Google Cloud
http://cs231n.github.io/gce-tutorial/
http://cs231n.github.io/gce-tutorial-gpus/

# Resources
## TensorFlow
git clone https://github.com/leriomaggio/deep-learning-keras-tensorflow

* https://github.com/anishathalye/neural-style

ModelZoo
* https://github.com/tensorflow/models

## TensorBoard 
* https://www.tensorflow.org/get_started/graph_viz
* https://www.tensorflow.org/get_started/summaries_and_tensorboard

## Lasagne
* Neural Art Style Transfer
  https://github.com/Lasagne/Recipes/blob/master/examples/styletransfer/Art%20Style%20Transfer.ipynb
* KFKD
  http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/
  https://www.kaggle.com/c/facial-keypoints-detection/details/deep-learning-tutorial

## dist-Keras Keras+Spark
https://db-blog.web.cern.ch/blog/joeri-hermans/2017-01-distributed-deep-learning-apache-spark-and-keras

## elephas Keras+Spark
https://github.com/maxpumperla/elephas

# Courses
* CS231N Convolutional NN for Visual Recognition
* CS224D Deep Learning for Natural Language Processing
* http://cvit.iiit.ac.in/summerschool/resources.html
* Deep RL
  * CS294 Deep Reinforcement Learning
  * http://efavdb.com/battleship/
  * http://karpathy.github.io/2016/05/31/rl/
* DeepLearning Course By Google
  * https://www.udacity.com/course/deep-learning--ud730
