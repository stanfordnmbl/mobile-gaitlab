# GPU version
# docker run --gpus all -it --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v $(pwd)/in:/gaitlab/input -v $(pwd)/out:/gaitlab/output stanfordnmbl/video-cp /bin/sh -c 'python3 demo.py'

# CPU version
docker run -v $(pwd)/in:/gaitlab/input -v $(pwd)/out:/gaitlab/output stanfordnmbl/video-cp /bin/sh -c 'python3 demo.py'
