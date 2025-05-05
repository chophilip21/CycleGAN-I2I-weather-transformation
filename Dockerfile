# Use the official PyTorch image as base
# Use the official PyTorch image as base
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

# Set the working directory inside the container
WORKDIR /workspace

# Copy the entire project into the container
COPY . /workspace

# Install Python dependencies
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install setuptools wheel

# Run the setup script
RUN make setup

# Set environment variables to prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Expose the port for any future services or applications (optional)
EXPOSE 8888

# The default command to run when the container starts (optional)
CMD ["bash"]
