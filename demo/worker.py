from celery import Celery
from demo import predict
import urllib.request
import shutil
import os
import requests
import json

app = Celery('gaitlab', broker='redis://redis:6379/0')

@app.task(name='gaitlab.cp')
def cp(args):
    path = "/gaitlab/input/input.mp4"

    # remove the old file
    os.system('rm {}'.format(path))

    # save the new file
    url = args["video_url"]
    with urllib.request.urlopen(url) as response, open(path, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)

    # run predictions
    result, video = predict(path)

    files = {'file': video}

    # store results
    r = requests.post("http://www/annotation/{}/".format(args["annotation_id"]),
                      files = files,
                      data = {"result": json.dumps(result)})

    return None
