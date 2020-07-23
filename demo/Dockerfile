FROM cwaffles/openpose
RUN apt update ; apt install ffmpeg python3-setuptools -y
COPY requirements.txt /gaitlab/requirements.txt
WORKDIR /gaitlab
RUN pip3 install -r requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -U tensorflow-gpu
COPY . /gaitlab
