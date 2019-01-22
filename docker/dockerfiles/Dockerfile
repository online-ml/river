FROM python:3.6.8-jessie

RUN mkdir /app

WORKDIR /app


RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


RUN python -m pip install -U numpy
# Clone scikit-multiflow
RUN git clone https://github.com/scikit-multiflow/scikit-multiflow.git
# Install scikit-multiflow
RUN cd scikit-multiflow && python -m pip install -U . && cd .. && rm -rf scikit-multiflow

COPY /examples/src /app
COPY /examples/data /app

CMD /bin/bash

