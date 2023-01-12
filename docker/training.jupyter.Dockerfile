FROM pitch-geo-training

WORKDIR /workspace
COPY docker/training_jupyter_requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt
RUN rm requirements.txt

ENTRYPOINT ["jupyter-lab", "--notebook-dir", "/workspace/notebooks"]