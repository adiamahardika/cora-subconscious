from flask import render_template, Response
from app.utils import response 
import app.object_detection as object_detection

def index():
    return render_template('frontend.html')

def feed():
    return Response(object_detection.start_inference(), mimetype='multipart/x-mixed-replace; boundary=frame')
