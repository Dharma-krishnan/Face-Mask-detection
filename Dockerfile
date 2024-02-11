# Use an official Python runtime as the base image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Add the current directory to the container
ADD . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    wget \
    libboost-all-dev \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    gfortran \
    openexr \
    libatlas-base-dev \
    libopenblas-dev \
    liblapack-dev \
    libblas-dev \
    libhdf5-dev \
    libhdf5-serial-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set up environment variables
ENV STREAMLIT_SERVER_URL=http://localhost:8501 \
    STREAMLIT_RUN_ON_SAVE=true \
    GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json

# Expose the Streamlit app port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py"]