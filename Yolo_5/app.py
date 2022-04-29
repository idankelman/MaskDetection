from flask import Flask, render_template, Response
from camera import Video
import os
import cv2
app=Flask(__name__)

def gen(camera):
    while True:
        frame=camera.get_frame()
        if frame is not None:
            yield(b'--frame\r\n'
            b'Content-Type:  image/jpeg\r\n\r\n' + frame +
                b'\r\n\r\n')

@app.route('/video')

def video():
    return Response(gen(Video()),
    mimetype='multipart/x-mixed-replace; boundary=frame')

app.run(host="localhost", port=5000, debug=True)