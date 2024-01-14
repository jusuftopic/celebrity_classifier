from flask import Flask, request, jsonify
import classifier_utils

app = Flask(__name__)


@app.route('/classify_image', methods=['GET', 'POST'])
def classify_image():
    image = request.form['image_data']

    result = jsonify(classifier_utils.classify_image(image))
    result.headers.add('Access-Control-Allow-Origin', '*')

    return result


if __name__ == '__main__':
    classifier_utils.load_classifier()
    classifier_utils.load_model()
    port = 8000
    print("Server started on the port: " + str(port))
    app.run(port=port)