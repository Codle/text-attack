# Base Image
FROM registry.cn-shanghai.aliyuncs.com/codle/text-attack-git:0.1

ADD . /

WORKDIR /

# RUN pip install -i https://mirrors.aliyun.com/pypi/simple editdistance
# RUN pip install -i https://mirrors.aliyun.com/pypi/simple -r requirements.txt

CMD ["sh", "run.sh"]
