# Docker images

There are a lot of components. Best way to deal with dependencies is to use docker.

```bash

# build
docker build -t weather-gan .

# run
docker run --gpus all -it --rm \
  -v "$(pwd)":/workspace \
  --gpus device=0 \
  --shm-size 64G \
  weather-gan:latest
```