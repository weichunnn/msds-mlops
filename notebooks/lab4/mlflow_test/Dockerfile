FROM python:3.9-slim

WORKDIR /app

RUN pip install mlflow==2.8.1 \
    scikit-learn \
    pandas \
    numpy

EXPOSE 5000

ENV MLFLOW_TRACKING_URI=http://localhost:5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]