


FROM ubuntu:18.04
#LABEL maintainer="Walid Gara"
RUN mkdir /app

WORKDIR /app

# Install some dependencies
#RUN apt-get update && apt-get install -y --no-install-recommends \
#        build-essential \
#        curl \
#        libfreetype6-dev \
#        libhdf5-serial-dev \
#        libpng-dev \
#        libzmq3-dev \
#        pkg-config \
#        python \
#        python-dev \
#        rsync \
#        software-properties-common \
#        unzip \
#        && \
#    apt-get clean && \
#    rm -rf /var/lib/apt/lists/*

ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.6 \
        curl \
        ca-certificates \
        build-essential \
        python3.6-dev \
        python3-distutils \
        git \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Install pip
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3.6 get-pip.py && \
    rm get-pip.py


RUN python3.6 -m pip install -U numpy

# Install Jupyter
RUN python3.6 -m pip install -U jupyter

# Clone scikit-multiflow
RUN git clone https://github.com/scikit-multiflow/scikit-multiflow.git
# Install scikit-multiflow
RUN cd scikit-multiflow && python3.6 -m pip install -U . && cd .. && rm -rf scikit-multiflow

# Copy QuickStart Notebook
COPY /examples/notebooks /app
COPY /examples/data /app

# Jupyter port
EXPOSE 8888

# Start jupyter notebook
CMD jupyter notebook --ip 0.0.0.0 --no-browser --allow-root