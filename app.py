from flask import Flask, render_template, request, send_file
import numpy as np
from scipy.special import eval_genlaguerre
import matplotlib.pyplot as plt
from PIL import Image
import os
from io import BytesIO
import base64

app = Flask(__name__)

# Temporary files for saving the generated images
TEMP_INTENSITY_FILE = "temp_intensity.bmp"
TEMP_PHASE_FILE = "temp_phase.bmp"
TEMP_CROPPED_INTENSITY_FILE = "temp_cropped_intensity.bmp"
TEMP_CROPPED_PHASE_FILE = "temp_cropped_phase.bmp"

def generate_asymmetric_lg_beam(l, p, j, beam_size_mm, delta, grid_size=2500):
    """Generate a 2500x2500 LG beam"""
    omega_0 = beam_size_mm / 2
    grid_extent = beam_size_mm
    x = np.linspace(-grid_extent / 2, grid_extent / 2, grid_size)
    y = np.linspace(-grid_extent / 2, grid_extent / 2, grid_size)
    X, Y = np.meshgrid(x, y)
    zeta, eta = -1, 1
    chi = (X + zeta * delta) + 1j * (Y + eta * delta)
    rho_squared = X**2 + Y**2
    gaussian_envelope = np.exp(-rho_squared / omega_0**2)
    amplitude = (np.sqrt(2) * chi / omega_0) ** np.abs(l)
    n = p - j
    m = np.abs(l + j)
    if n < 0 or m < 0:
        raise ValueError("Invalid parameters: p - j >= 0 and |l + j| >= 0 required.")
    radial_term = eval_genlaguerre(n, m, 2 * rho_squared / omega_0**2)
    field = amplitude * radial_term * gaussian_envelope
    intensity_pattern = np.abs(field) ** 2
    phase_pattern = np.angle(field)
    return intensity_pattern, phase_pattern

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    try:
        # Get form inputs
        l = int(request.form["l"])
        p = int(request.form["p"])
        j = int(request.form["j"])
        beam_size_mm = float(request.form["beam_size"])
        delta = float(request.form["delta"])
        crop_size_x = int(request.form["crop_x"])  # Pixel units
        crop_size_y = int(request.form["crop_y"])  # Pixel units

        # Generate 2500x2500 LG beam
        intensity, phase = generate_asymmetric_lg_beam(l, p, j, beam_size_mm, delta)

        # Save the full 2500x2500 image temporarily
        save_as_bmp(intensity, TEMP_INTENSITY_FILE, normalize=True)
        save_as_bmp(phase, TEMP_PHASE_FILE, normalize=False)

        # Calculate cropping indices
        grid_size = intensity.shape[0]

        # Crop intensity and phase using pixel-based cropping
        crop_x_min = int((grid_size / 2) - (crop_size_x / 2))
        crop_x_max = int((grid_size / 2) + (crop_size_x / 2))
        crop_y_min = int((grid_size / 2) - (crop_size_y / 2))
        crop_y_max = int((grid_size / 2) + (crop_size_y / 2))

        # Ensure cropping indices are within bounds
        crop_x_min = max(crop_x_min, 0)
        crop_x_max = min(crop_x_max, grid_size)
        crop_y_min = max(crop_y_min, 0)
        crop_y_max = min(crop_y_max, grid_size)

        # Crop intensity and phase
        intensity_cropped = intensity[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        phase_cropped = phase[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

        # Save cropped intensity and phase as BMP files
        save_as_bmp(intensity_cropped, TEMP_CROPPED_INTENSITY_FILE, normalize=True)
        save_as_bmp(phase_cropped, TEMP_CROPPED_PHASE_FILE, normalize=False)

        # Prepare cropped plot for display
        buffer = BytesIO()
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(intensity_cropped, cmap="hot", origin="lower")
        ax[0].set_title("Cropped Intensity")
        ax[1].imshow(phase_cropped, cmap="hsv", origin="lower")
        ax[1].set_title("Cropped Phase")
        plt.tight_layout()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        img_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
        buffer.close()

        # Return the generated images and the form inputs
        return render_template("results.html", img_data=img_data, l=l, p=p, j=j, beam_size=beam_size_mm, delta=delta, crop_x=crop_size_x, crop_y=crop_size_y)
    except Exception as e:
        return f"Error: {str(e)}"

def save_as_bmp(data, filename, normalize=True):
    """Save numpy array as BMP"""
    if normalize:
        data = (data / np.max(data) * 255).astype(np.uint8)  # Normalize to 0-255
    else:
        data = ((data + np.pi) / (2 * np.pi) * 255).astype(np.uint8)  # Map phase to 0-255
    image = Image.fromarray(data)
    image.save(filename)

@app.route("/download/<file_type>")
def download(file_type):
    """Download BMP file for intensity or phase"""
    try:
        if file_type == "intensity":
            return send_file(TEMP_CROPPED_INTENSITY_FILE, as_attachment=True, download_name="intensity_pattern.bmp")
        elif file_type == "phase":
            return send_file(TEMP_CROPPED_PHASE_FILE, as_attachment=True, download_name="phase_pattern.bmp")
        else:
            return "Invalid file type", 400
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
