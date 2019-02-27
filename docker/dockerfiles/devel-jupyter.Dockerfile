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
RUN cd scikit-multiflow && python3.6 -m pip install -U . && cd ..

# Copy QuickStart notebook
RUN cp scikit-multiflow/docker/examples/notebooks/* .
RUN cp scikit-multiflow/docker/examples/data/* .

# Download elec dataset
RUN python3.6 data.py


# Clean directory
RUN rm -rf scikit-multiflow
RUN rm data.py

# Jupyter port
EXPOSE 8888

# Start jupyter notebook
CMD jupyter notebook --ip 0.0.0.0 --no-browser --allow-root