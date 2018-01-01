# AWS Reminders

Dashboard:
https://eu-west-2.console.aws.amazon.com/ec2/v2/home?region=eu-west-2

Launch a Free Tier instance, and get address from the dashboard. Log in via ssh with the command below.

Terminate from dashboard by selecting instance and then:
Actions -> Instance State -> Terminate

Check from dashboard that all instances are terminated correctly.

## awscli

### Starting instances

* Ubuntu:
```bash
aws ec2 run-instances --image-id ami-fcc4db98 --count 1 --instance-type t2.micro --key-name AWS-London --security-group-ids sg-47c9562f

ssh -i ~/.ssh/AWS-London.pem ubuntu@XXX
```

* Official:
```bash
aws ec2 run-instances --image-id ami-e7d6c983 --count 1 --instance-type t2.micro --key-name AWS-London --security-group-ids sg-47c9562f

ssh -i ~/.ssh/AWS-London.pem ec2-user@XXX
```

### Querying and terminating instances

```bash
aws ec2 describe-instances

aws ec2 describe-instances --instance-ids i-XXX  --query 'Reservations[0].Instances[0].PublicIpAddress'

aws ec2 describe-instances --query 'Reservations[*].Instances[*].[InstanceId]' --filters Name=instance-state-name,Values=running --output text

aws ec2 terminate-instances --instance-ids i-XXX
```




#### eof

