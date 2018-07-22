from flask import Flask, request, render_template
from PIL import Image
from io import BytesIO
import base64
import urllib
from filter import add_dog_filter, get_filter
import numpy as np


app = Flask(__name__)

dog_filter = get_filter()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/submit',methods=["POST"])
def submit_file():
    #img = request.files['imagefile']
    img = Image.open(BytesIO(request.files['imagefile'].read()))
    imx = np.array(img)

    result = add_dog_filter(imx, dog_filter)

    res_img = Image.fromarray(result)

    inp_byte_io = BytesIO()
    img.save(inp_byte_io, 'PNG')
    inp_byte_io.seek(0)

    res_byte_io = BytesIO()
    res_img.save(res_byte_io, 'PNG')
    res_byte_io.seek(0)

    inp_png_output = base64.b64encode(inp_byte_io.getvalue())
    input_processed_file = urllib.parse.quote(inp_png_output)

    res_png_output = base64.b64encode(res_byte_io.getvalue())
    res_processed_file = urllib.parse.quote(res_png_output)


    return render_template('result.html', inp_processed_file = input_processed_file, res_processed_file = res_processed_file)


if __name__=='__main__':
    app.run(threaded=True, debug=True) 