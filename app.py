from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
from utils.face_helper import detect_face

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():

    image = request.files["image"]

    image.save("capture.png")

    face = detect_face("capture.png")

    if face is None:

        return jsonify({
            "emotion": "Không phát hiện khuôn mặt",
            "confidence": 0
        })

    return jsonify({
        "emotion": "Đã phát hiện khuôn mặt",
        "confidence": 100
    })
if __name__ == '__main__':
    app.run(debug=True)