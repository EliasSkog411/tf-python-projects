# Convert to CoreML
import coremltools as ct
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


model = ct.convert("tensorflow_models/ep_48_model_savemodel", convert_to="mlprogram" ,inputs=[ct.ImageType(color_layout=ct.colorlayout.BGR, bias=[-1,-1,-1], scale=1/127.5 )], source='tensorflow')


# Use PIL to load and resize the image to expected size.
from PIL import Image
example_image = Image.open("daisy.jpg").resize((512, 512))

spec = model.get_spec()
print(spec.description)

# Make a prediction using Core ML.p
out_dict = model.predict({"LogVisionHigh": example_image})

# print info about inputs and outputs
print(spec.description)

model.save('new_models/placeholder.mlpackage')