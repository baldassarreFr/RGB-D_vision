# Build:  docker build -t rgbd-vision .
FROM python:3.11-slim

# System libraries Open3D needs on Linux
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libgomp1 libegl1 libusb-1.0-0 git openssh-client \
 && rm -rf /var/lib/apt/lists/*

# Install the Python packages
WORKDIR /work
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install the project's own code (src/rgbd_sim) as a package
COPY . .
RUN pip install --no-cache-dir -e .

# No screen inside the container: matplotlib draws to files instead of windows
ENV MPLBACKEND=Agg
ENV OMP_NUM_THREADS=1


CMD ["python", "checks/check_sample.py"]