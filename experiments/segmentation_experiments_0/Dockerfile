# TensorFlow GPU base image
FROM tensorflow/tensorflow:latest-gpu

# Set working directory
WORKDIR /tf-knugs

# Install system dependencies required by OpenCV
USER root
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install -U jupyterlab pandas matplotlib opencv-python scikit-learn pillow

# Expose Jupyter port
EXPOSE 8888

# Default entrypoint: start JupyterLab
ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]

# Switch back to non-root user if desired
# USER jupyter

