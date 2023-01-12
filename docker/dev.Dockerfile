FROM pitch-geo-training

WORKDIR /workspace
COPY docker/dev_requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt
RUN rm requirements.txt