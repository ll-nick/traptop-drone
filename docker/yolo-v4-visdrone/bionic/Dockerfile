FROM wsascha/darknet:18.04

RUN pip install setuptools
RUN pip install --upgrade setuptools pip
RUN pip install click opencv-python

COPY object_detection.py net.cfg net.data net.names net.weights ./
ENTRYPOINT ["./object_detection.py", "net.cfg", "net.data", "net.weights", "/data/inputs", "--save_to", "/data/outputs"]
