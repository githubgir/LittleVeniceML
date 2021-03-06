# Spark Clusters
Setup instructions for AMI with Spark and ML libraries
* http://christopher5106.github.io/big/data/2016/01/27/two-AMI-to-create-the-fastest-cluster-with-gpu-at-the-minimal-engineering-cost-with-EC2-NVIDIA-Spark-and-BIDMach.html
* https://github.com/amplab/spark-ec2.git
* https://github.com/christopher5106/spark-ec2
* http://ampcamp.berkeley.edu/3/exercises/launching-a-bdas-cluster-on-ec2.html


## Use AWS CLI to launch an instance
List of AMIs supported by spark e.g. ami-72320f37 in 

https://github.com/amplab/spark-ec2/blob/v4/ami-list/us-west-1/hvm

For some reason I cannot fine the needed AMI via EC2 dashboard, therefore need to launch via AWS CLI, 
see [AWS Launch via CLI](https://github.com/githubgir/LittleVeniceML/blob/master/AWS_LAUNCH_CLI.md)

## Launch Cluster from EC2
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
