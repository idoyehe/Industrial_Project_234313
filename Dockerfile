# Extracted from: https://github.com/ibm-functions/runtime-python/tree/master/python3

FROM python:3.5-slim-jessie

ENV FLASK_PROXY_PORT 8080

RUN apt-get update && apt-get install -y \
        wget \        
        build-essential cmake pkg-config \       
        gcc \
        libc-dev \
        libxslt-dev \
        libxml2-dev \
        libffi-dev \
        libssl-dev \
        zip \
        unzip \
        vim \
        cmake \
        && rm -rf /var/lib/apt/lists/*

RUN apt-cache search linux-headers-generic

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip setuptools six && pip install --no-cache-dir -r requirements.txt
RUN pip install opencv-contrib-python-headless
RUN pip install opencv-python-headless
RUN pip install dlib
RUN wget https://github.com/cmusatyalab/openface/archive/0.2.1.tar.gz
RUN tar -zxvf 0.2.1.tar.gz
WORKDIR "/openface-0.2.1"
#RUN git clone https://github.com/cmusatyalab/openface.git
#WORKDIR "/openface"
RUN python setup.py install

RUN mkdir -p /actionProxy
ADD https://raw.githubusercontent.com/apache/incubator-openwhisk-runtime-docker/dockerskeleton%401.12.0-incubating/core/actionProxy/actionproxy.py /actionProxy/actionproxy.py

RUN mkdir -p /pythonAction
ADD https://raw.githubusercontent.com/apache/incubator-openwhisk-runtime-python/3%401.0.3/core/pythonAction/pythonrunner.py /pythonAction/pythonrunner.py

CMD ["/bin/bash", "-c", "cd /pythonAction && python -u pythonrunner.py"]
