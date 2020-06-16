import base64
from io import BytesIO
from PIL import Image

def convert_and_save(b64_string):
    with open("imageToSave.jpg", "wb") as fh:
        fh.write(base64.decodebytes(b64_string.encode()))

def save_captured_image(file, image_name):
    starter = file.find(',')
    image_data = file[starter+1:]
    image_data = bytes(image_data, encoding="ascii")
    im = Image.open(BytesIO(base64.b64decode(image_data)))
    cap_image_path = 'static/captured/'+image_name+'.jpg'
    im.save(cap_image_path)
    return cap_image_path
