FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir flask transformers peft

EXPOSE 5000

CMD ["python", "app.py"]
