# Specify base image
FROM python:3.7.2

# Install Python modules
RUN pip install -r requirements.txt

# Create new directory for running files
COPY . /code