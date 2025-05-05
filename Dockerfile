# Use the NVIDIA PyTorch 24.05 image (includes PyTorch, TorchVision & Apex AMP, Dali)
FROM nvcr.io/nvidia/pytorch:24.05-py3

# Set the working directory inside the container
WORKDIR /workspace

# Copy only your project files; data/env/samples should be mounted at runtime
COPY . /workspace

# Upgrade pip and install basic packaging tools
RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install setuptools wheel

# Install your project's development dependencies
RUN python3 -m pip install --upgrade -e .[devel]

# Install mseg-api directly from GitHub
RUN python3 -m pip install git+https://github.com/mseg-dataset/mseg-api.git

# Prevent Python from writing .pyc files and ensure unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# (Optional) Expose a port for services like Jupyter
EXPOSE 8888

# Default to bash
CMD ["bash"]
