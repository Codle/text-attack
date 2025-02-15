# Base Image
FROM registry.cn-shanghai.aliyuncs.com/codle/text-attack-git:0.1.4

ADD . /

WORKDIR /

RUN wget -P /data/ https://ai.tencent.com/ailab/nlp/data/Tencent_AILab_ChineseEmbedding.tar.gz -q
RUN tar -xvf /data/Tencent_AILab_ChineseEmbedding.tar.gz -C /data/
RUN head /data/Tencent_AILab_ChineseEmbedding.txt -n 100001 > /data/tencent.txt
RUN rm /data/Tencent_AILab_ChineseEmbedding.tar.gz
RUN rm /data/Tencent_AILab_ChineseEmbedding.txt

RUN ls /data/ -al

# RUN pip install -i https://mirrors.aliyun.com/pypi/simple editdistance
# RUN pip install -i https://mirrors.aliyun.com/pypi/simple -r requirements.txt

CMD ["sh", "run.sh"]
