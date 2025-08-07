# FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

# WORKDIR /app


FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

# Set working directory
WORKDIR /app

# Install system dependencies (optional but good practice)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --upgrade pip
RUN pip install \
    transformers==4.41.2 \
    torch \
    numpy \
    streamlit \
    fastapi \
    uvicorn

# Set working directory
WORKDIR /app
# # Copy your application code into the container
# COPY . /app

# # Expose Streamlit port (if GUI)
# EXPOSE 8501

# # Set the default command (can override in docker-compose or CLI)
# CMD ["streamlit", "run", "main.py"]
