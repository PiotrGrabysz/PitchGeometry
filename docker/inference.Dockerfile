FROM pitch-geo-tensorflow

WORKDIR workspace
COPY docker/inference_requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt
RUN rm requirements.txt

ENTRYPOINT ["python3", "pitch_geo/inference/inference.py"]