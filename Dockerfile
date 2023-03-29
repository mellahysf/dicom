FROM ubuntu:20.04
RUN apt-get update && apt-get -y update
RUN apt-get -y update \
    && apt-get install -y wget \
    && apt-get install -y jq \
    && apt-get install -y lsb-release \
    && apt-get install -y openjdk-8-jdk-headless \
    && apt-get install -y build-essential python3-pip \
    && pip3 -q install pip --upgrade \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
            /usr/share/man /usr/share/doc /usr/share/doc-base

RUN apt-get update && apt-get install libgl1
RUN apt update
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6

COPY ./ app
RUN pip install -r /app/requirements.txt

RUN chmod +x app/main.sh


ENTRYPOINT ["/app/main.sh"]
