FROM ubuntu:22.04
RUN apt update && apt upgrade -y
RUN apt-get install -y python3 python3-pip
RUN apt install -y wget

RUN python3 -m  pip install numpy pandas river matplotlib
# Any working directory can be chosen as per choice like '/' or '/home' etc
# i have chosen /usr/app/src
WORKDIR /usr/app/src

#to COPY the remote file at working directory in container
COPY ./datasets/ ./datasets/
COPY ./evaluation/ ./evaluation/
COPY ./methods/ ./methods/
COPY ./metrics/ ./metrics/
COPY test.py ./
# Now the structure looks like this '/usr/app/src/test.py'


#CMD instruction should be used to run the software
#contained by your image, along with any arguments.

CMD [ "python3", "./test.py"]