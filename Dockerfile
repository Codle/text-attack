# Base Image
FROM registry.cn-shanghai.aliyuncs.com/codle/text-attack:0.2

ADD . /

WORKDIR /

# RUN pip install -i https://mirrors.aliyun.com/pypi/simple -r requirements.txt

CMD ["sh", "run.sh"]
