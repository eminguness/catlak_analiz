from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import subprocess

app = Flask(__name__, static_folder="static", template_folder="templates")

@app.route('/analiz')
def analiz():
    return render_template('index.html')

@app.route('/')
def home():
    return render_template('anasayfa.html')

@app.route('/hakkimizda')
def hakkimizda():
    return render_template('hakkimizda.html')

@app.route('/iletisim')
def iletisim():
    return render_template('iletisim.html')
@app.route('/detect', methods=['POST'])
def detect():
    uploaded_file = request.files.get('file')
    if not uploaded_file:
        return jsonify({"error": "No file uploaded"}), 400

    temp_dir = "tempDir"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    img_path = os.path.join(temp_dir, uploaded_file.filename)
    uploaded_file.save(img_path)

    detect_command = [
        "python", "yolov7/detect.py",
        "--weights", "weights/best.pt",
        "--source", img_path,
        "--device", "cpu"
    ]
    try:
        subprocess.run(detect_command, check=True)
    except subprocess.CalledProcessError as e:
        return jsonify({"error": "Model execution failed", "details": str(e)}), 500

    output_dir = "runs/detect"
    all_dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    if not all_dirs:
        return jsonify({"error": "No output directory found"}), 500

    latest_exp_dir = sorted(all_dirs, key=lambda x: int(x.replace("exp", "")) if x.replace("exp", "").isdigit() else -1, reverse=True)[0]
    result_img_path = os.path.join(output_dir, latest_exp_dir, os.path.basename(img_path))

    if os.path.exists(result_img_path):
        relative_path = os.path.relpath(result_img_path, start="runs/detect")
        return jsonify({"success": True, "result_image": f"/runs/detect/{relative_path}"}), 200
    else:
        return jsonify({"error": "Result image not found"}), 500


@app.route('/runs/detect/<path:filename>')
def serve_detected_image(filename):
    return send_from_directory('runs/detect', filename)


if __name__ == '__main__':
    app.run(debug=True)
