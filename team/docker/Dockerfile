# Submission image. Build from the repo root:
#   docker build -f team/docker/Dockerfile -t ecm27-policy:latest .
# For a GPU model, switch the base to e.g. nvidia/cuda:12.x-runtime + python3.12 and add torch.
FROM python:3.12-slim

WORKDIR /app
RUN python -m pip install --no-cache-dir grpcio==1.85.3 protobuf==7.36.2 numpy==2.3.1

COPY robofinals/__init__.py /app/robofinals/__init__.py
COPY robofinals/distributed/chamber_schema.py /app/robofinals/distributed/chamber_schema.py
COPY robofinals/distributed/grpc_policy.py /app/robofinals/distributed/grpc_policy.py
COPY robofinals/distributed/policy_v1_pb2.py /app/robofinals/distributed/policy_v1_pb2.py
COPY team/__init__.py /app/team/__init__.py
COPY team/policy/ /app/team/policy/
# COPY checkpoints/ /app/checkpoints/     # model weights (also allow-list them in Dockerfile.dockerignore)

ENV PYTHONPATH=/app
ENTRYPOINT ["python", "-u", "-m", "team.policy.client"]
