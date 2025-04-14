FROM pytorch/pytorch:latest

WORKDIR /app

COPY requirements.txt .
COPY training/train.py train.py
COPY data/inference.csv data/inference.csv
COPY data/train.csv data/train.csv
COPY model model
COPY . .

RUN pip install -r requirements.txt

CMD ["python", "train.py"]