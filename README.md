<<<<<<< HEAD
# MLE
=======
# Iris Classification with Docker

## Project Structure
- `data/`: Contains training and inference datasets.
- `training/`: Training-related scripts and Dockerfile.
- `inference/`: Inference-related scripts and Dockerfile.
- `src/`: Data processing script.
- `requirements.txt`: Dependencies for the project.

## Steps to Run

1. Clone the repository.

2. Install Docker.

### Training

To train the model:

1. Navigate to the `iris-classification-docker` directory.
2. Build the Docker image:

docker build -t charan/train -f training/Dockerfile .

3. Run the training container:

docker run charan/train


This will train the model and save it to `/app/model.pth`.

### Inference

To perform inference:

1. Navigate to the `iris-classsification-docker` directory.
2. Build the Docker image:

docker build -t charan/inference -f inference/Dockerfile .

3. Run the inference container:

docker run charan/inference


This will generate predictions and save them to `data/predictions.csv`.

### Testing

Run tests with `pytest`:

pytest tests/test_train.py
pytest tests/test_inference.py



This is the kind of output you will obtain after running the tests.
>>>>>>> 2f18158 (comm)
