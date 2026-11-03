FROM python:3.11-slim

# System libraries Open3D needs on Linux (OpenGL and OpenMP runtimes)
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libgomp1 libegl1 libusb-1.0-0 git \
 && rm -rf /var/lib/apt/lists/*
# Install the Python packages
WORKDIR /work
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# No screen inside the container: matplotlib draws to files instead of windows
ENV MPLBACKEND=Agg


CMD ["python", "depth_first.py"]