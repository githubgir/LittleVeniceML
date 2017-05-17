# AWS CLI Command Line Interface
Automated launch of EC2 instance on windows including launching
* 3x SSH connections
* Chrome with Jupyter Notebook link
* Chrome with TensorBoard link

## AWS Configure
It requires aws cli installed and configured
http://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html
```
aws configure
# AWS Access Key ID [****************HVHA]:
# AWS Secret Access Key [****************mfU3]:
# Default region name [us-west-1]:
# Default output format [json]:
```

## Create Instance
```
SET AWS_AMI=ami-72320f37
aws ec2 describe-images --image-ids %AWS_AMI%

aws ec2 run-instances --image-id %AWS_AMI% --instance-type m4.large --key-name AWSKeyPair2 --security-groups sec-grp-ssh-jupyter-spark
#or
aws ec2 request-spot-instances --spot-price "0.1" --launch-specification file://C:/Dev/AWS/spot.json

aws ec2 describe-instances --query "Reservations[*].Instances[*].[InstanceId, ImageId, PublicDnsName]"
aws ec2 describe-instances --query "Reservations[*].Instances[*].[InstanceId,ImageId,PublicDnsName]" --filters "Name=image-id,Values=%AWS_AMI%"

set AWS_URL=PublicDnsName
```

## Launch SSH
```
set AWS_URL=ec2-54-193-106-52.us-west-1.compute.amazonaws.com
set AWS_USER=ec2-user
set AWS_USER=ubuntu

# run jupyter notebook here
putty -ssh -i C:\Dev\AWS\AWSKeyPair2.ppk %AWS_USER%@%AWS_URL%
# run nvidia-smi -l 1
putty -ssh -i C:\Dev\AWS\AWSKeyPair2.ppk %AWS_USER%@%AWS_URL%
# run tensorboard --logdir=/tmp/tf_logs
putty -ssh -i C:\Dev\AWS\AWSKeyPair2.ppk %AWS_USER%@%AWS_URL%
# just in case
putty -ssh -i C:\Dev\AWS\AWSKeyPair2.ppk %AWS_USER%@%AWS_URL%

pscp -i C:\Dev\AWS\AWSKeyPair2.ppk C:\Dev\AWS\AWSKeyPair2.pem %AWS_USER%@%AWS_URL%:/home/%AWS_USER%/AWSKeyPair2.pem

"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" "https:\\%AWS_URL%:8888"
"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" "http:\\%AWS_URL%:6006"
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
