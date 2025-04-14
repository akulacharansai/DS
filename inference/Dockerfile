FROM pytorch/pytorch:latest

# Set the working directory
WORKDIR /app

# Copy necessary files
COPY requirements.txt requirements.txt
COPY training/train.py train.py
COPY inference/inference.py inference.py
COPY data/train.csv data/train.csv
COPY data/inference.csv inference.csv
COPY model/model.pth model.pth

COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Run inference
CMD ["python", "inference.py"]