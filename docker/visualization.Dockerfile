FROM python:3.8.10-slim-buster

RUN python3 -m pip install -U pip
WORKDIR workspace
COPY docker/visualization_requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt
RUN rm requirements.txt

ENV PYTHONPATH="/workspace"

ENTRYPOINT ["python3", "pitch_geo/show_keypoints.py"]