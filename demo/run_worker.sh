#docker build . -t stanfordnmbl/video-cp
docker run --link gaitlab_redis_1:redis --link gaitlab_www_1:www --net gaitlab_default --gpus all -d --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v $(pwd)/in:/gaitlab/input -v $(pwd)/out:/gaitlab/output stanfordnmbl/video-cp /bin/sh -c 'celery -A worker worker --loglevel=info --concurrency=1 --pool=solo'
