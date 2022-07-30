from distutils.log import debug
from flask import Flask, render_template, request
from classifier import SnakeClassification
app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    image_file = request.files['file']
    image_path = "./images/" +image_file.filename
    image_file.save(image_path)
    result = SnakeClassification(image_path,'_model_8.pt').predict()

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(port=3000, debug=True)