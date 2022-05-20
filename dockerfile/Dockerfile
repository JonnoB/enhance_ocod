FROM python:3.8-slim-buster

# Set bash as default shell
#ENV SHELL=/bin/bash

# Create a working directory
#RUN mkdir /app
#WORKDIR /app

#upgrade pip and install dependencies
RUN python -m pip install --upgrade pip && \
	pip install -U pip setuptools wheel && \
	pip install numpy && \
	pip install pandas && \
	pip install spacy && \
	python -m spacy download en_core_web_lg
	


ENTRYPOINT [ "python" ]

EXPOSE 8888

#docker run --rm -it -v $(pwd):/app jonno/parse_process:test ./app/enhance_ocod/full_ocod_parse_process.py ./app/data/
