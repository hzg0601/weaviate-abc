FROM semitechnologies/transformers-inference:custom
COPY ./bge-large-zh-v1.5 /app/models/model
ENV ENABLE_CUDA=1
RUN sed -i 's/AutoTokenizer.from_pretrained(model_path)/AutoTokenizer.from_pretrained(model_path,use_fast=False)/' /app/vectorizer.py

