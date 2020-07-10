FROM debian:buster-slim

#WORKDIR /home

COPY . .

RUN apt update -y\
	&& apt install bash gcc g++ python3 python3-pip -y\
	&& pip3 install -r requirements.txt\
	&& python3 setup.py install

ENTRYPOINT ["/scripts/run-pyross"]
