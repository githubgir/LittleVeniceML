# Launch on AWS (manual)

## Launch AWS EC2
Follow this tutorial but use our own shared AMIs (e.g. ami-e7321387)

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

## Jupyter Notebook
Adding Jupyter Notebook support, not required for our own AMI.
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
