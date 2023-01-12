FROM pitch-geo-tensorflow

WORKDIR /workspace
COPY docker/training_requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt
RUN rm requirements.txt