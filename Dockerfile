FROM debian:buster-slim

WORKDIR /home

COPY . ./pyross

RUN apt update -y\
	&& apt install bash gcc g++ -y\
	&& rm -rf pyross/examples\
	&& conda env create --file pyross/environment.yml

SHELL ["conda", "run", "-n", "pyross", "/bin/bash", "-c"]

RUN cd pyross/ \ 	
	&& python pyross/setup.py install 

ENV PATH /opt/conda/envs/pyross/bin:$PATH

ENTRYPOINT ["/bin/bash"]
