FROM python:3.7

ENV PYTHONUNBUFFERED 1
RUN apt-get update


RUN mkdir /usr/src/app
WORKDIR /usr/src/app
COPY requirements.txt /usr/src/app
RUN pip install -r requirements.txt
COPY . /usr/src/app/

CMD sh /usr/src/app/entrypoint.sh
 
EXPOSE 80
