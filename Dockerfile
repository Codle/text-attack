# Base Image
FROM registry.cn-shanghai.aliyuncs.com/codle/text-attack-git:1.2.3

ADD . /

WORKDIR /

RUN pip install -i https://mirrors.aliyun.com/pypi/simple editdistance
# RUN pip install -i https://mirrors.aliyun.com/pypi/simple -r requirements.txt

CMD ["sh", "run.sh"]
