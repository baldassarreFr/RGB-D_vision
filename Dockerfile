# Recipe for the project's Linux environment.
# Build:  docker build -t rgbd-synth .
# Run:    docker run --rm -v "${PWD}:/work" rgbd-synth            (PowerShell)

# 1. Start from a small official Debian image that already has Python 3.11
FROM python:3.11-slim

# 2. System libraries Open3D needs on Linux (OpenGL and OpenMP runtimes)
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# 3. Install the Python packages. Copying requirements.txt alone first means
#    Docker re-installs packages only when this file changes, not on every code edit.
WORKDIR /work
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. No screen inside the container: matplotlib draws to files instead of windows
ENV MPLBACKEND=Agg

# 5. Default command (override it by adding another command after the image name)
CMD ["python", "depth_first.py"]
