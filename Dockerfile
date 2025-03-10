# Use a lightweight Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the project files into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Streamlit port
EXPOSE 8501

# Set environment variable to indicate we are running in Docker
ENV DOCKERIZED=true

# Command to run the Streamlit app
CMD ["streamlit", "run", "src/app.py", "--server.enableCORS", "false", "--server.port=8501", "--server.address=0.0.0.0"]
