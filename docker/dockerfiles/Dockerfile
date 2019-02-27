FROM python:3.6.8-jessie

RUN mkdir /app

WORKDIR /app

RUN python -m pip install -U numpy

# Install scikit-multiflow
RUN python -m pip install -U scikit-multiflow

COPY /examples/src /app

COPY /examples/data /app

# Download elec dataset
RUN python data.py

RUN rm data.py

CMD /bin/bash
