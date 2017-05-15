# LittleVeniceML
Little Venice Machine Learning Group
# Content

Setup on AWS
Setup on AWS with spark-ec2
Lerio Maggio Tutorial

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

## NVIDIA
nvidia-smi -l 1

## Jupyter Notebook
http://efavdb.com/deep-learning-with-jupyter-on-aws/
* dont specify password otherwise not working
* dont specify certfile and key file otherwise not working on iPad
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

## SSH on iPad
http://www.packetnerd.com/?p=131


## Jupyter Notebook on iPad 
1) don't specify cerfiles in the script above, or
2) properly via Certificate Authority (own or 3rd party)

http://jupyter-notebook.readthedocs.io/en/latest/public_server.html

https://letsencrypt.org/getting-started/

## Spot instances greyed out for Market AMIs
https://www.paulwakeford.info/2016/01/07/aws-marketplace-and-spot-instances/
https://aws.amazon.com/marketplace/pp/B06VSPXKDX




## Lerio Maggio Tutorial
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
https://www.tensorflow.org/install/install_linux#InstallingAnaconda
https://www.tensorflow.org/install/install_linux#the_url_of_the_tensorflow_python_package

```
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.1.0-cp27-none-linux_x86_64.whl
#or
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.1.0-cp27-none-linux_x86_64.whl
#or
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.0.1-cp27-none-linux_x86_64.whl
```

## Upgrade NVIDIA drivers, CUDA, etc



# Resources
## TensorFlow
git clone https://github.com/leriomaggio/deep-learning-keras-tensorflow

https://github.com/anishathalye/neural-style
ModelZoo
https://github.com/tensorflow/models

## TensorBoard 
https://www.tensorflow.org/get_started/graph_viz
https://www.tensorflow.org/get_started/summaries_and_tensorboard

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


## Courses
* CS231N Convolutional NN for Visual Recognition
* CS224D Deep Learning for Natural Language Processing
* http://cvit.iiit.ac.in/summerschool/resources.html
* Deep RL
  * CS294 Deep Reinforcement Learning
  * http://efavdb.com/battleship/
  * http://karpathy.github.io/2016/05/31/rl/
* DeepLearning Course By Google
  * https://www.udacity.com/course/deep-learning--ud730


# Spark Clusters

## PySpark install and Jupyter notebook configuration
https://www.dataquest.io/blog/pyspark-installation-guide/
