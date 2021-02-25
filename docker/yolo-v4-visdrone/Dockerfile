FROM wsascha/darknet:base

COPY object_detection.py net.cfg net.data net.names net.weights ./
ENTRYPOINT ["./object_detection.py", "net.cfg", "net.data", "net.weights", "/data/inputs", "--save_to", "/data/outputs"]
