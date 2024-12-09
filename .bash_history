gcloud config set project <asymmetric-lg-beam>
mkdir lg_beam_webapp
cd lg_beam_webapp
nano app.py
from flask import Flask, render_template, request
import numpy as np
from scipy.special import eval_genlaguerre
import matplotlib.pyplot as plt
from io import BytesIO
import base64
app = Flask(__name__)
def generate_asymmetric_lg_beam(l, p, j, beam_size_mm, delta, grid_size=2000):
@app.route("/")
def index():
@app.route("/generate", methods=["POST"])
def generate():
if __name__ == "__main__":;     app.run(debug=True)
nano app.py
mkdir templates
nano templates/index.html
nano templates/results.html
pip install flask numpy scipy matplotlib
pip freeze > requirements.txt
python app.py
nano app.py
python app.py
pwd
ls
cd lg_beam_webapp
python app.py
nano app.py
python app.py
pwd
ls
cd /home/godthou0727/lg_beam_webapp
python app.py
app.run(debug=True)
app.run(host="0.0.0.0", port=8080, debug=True)
python app.py
gcloud app create --region=us-central
