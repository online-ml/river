FROM ubuntu:18.04

RUN mkdir /app

WORKDIR /app


ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.6 \
        curl \
        ca-certificates \
        build-essential \
        python3.6-dev \
        python3-distutils \
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

# Install scikit-multiflow
RUN python3.6 -m pip install -U scikit-multiflow

# Copy QuickStart Notebook
COPY /examples/notebooks /app
COPY /examples/data /app

# Download elec dataset
RUN python3.6 data.py

RUN rm data.py

# Jupyter port
EXPOSE 8888

# Start jupyter notebook
CMD jupyter notebook --ip 0.0.0.0 --no-browser --allow-root