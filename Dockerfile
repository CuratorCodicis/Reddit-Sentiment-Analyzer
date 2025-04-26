# Use a lightweight Python image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Install system dependencies for llama-cpp-python
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the project files into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directory for models and make it a volume
# Allows users to mount their model directory
RUN mkdir -p /app/models
VOLUME /app/models

# Expose the Streamlit port
EXPOSE 8501

# Set environment variable to indicate we are running in Docker
ENV DOCKERIZED=true

# Command to run the Streamlit app
CMD ["streamlit", "run", "src/app.py", "--server.enableCORS", "false", "--server.port=8501", "--server.address=0.0.0.0"]
