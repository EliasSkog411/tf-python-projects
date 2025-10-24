# Imports
import tensorflow as tf
import coremltools as ct
from PIL import Image

model_size = 512

# Step 1: Load the TensorFlow model manually using `SavedModel` loader
#model_path = "tensorflow_models/cs_512_100_old" # good cs model
#model_path = "tensorflow_models/ep_48_model_savemodel" # acceptable log 512
#model_path = "tensorflow_models/simple_256_50_old" # bad log 256
#model_path = "tensorflow_models/temp_fast_cs_permute"


model_path =  "tensorflow_models/VERY_GOOD_CS_PERMUTE"

loaded = tf.saved_model.load(model_path)
print(loaded.signatures.keys())

# Step 2: Choose one of the ConcreteFunctions (e.g., 'serving_default')
concrete_func = loaded.signatures["serving_default"]

# Step 3: Convert to CoreML using the selected function
coreml_model = ct.convert(
    [concrete_func],
    convert_to="mlprogram",  # Use Core ML's new model format
    source="tensorflow",
    inputs=[
        ct.ImageType(
            shape=(1,model_size, model_size, 3),
            name="keras_tensor",  # Adjust this name to match actual input key
            color_layout=ct.colorlayout.RGB,
            bias=[-1.0, -1.0, -1.0],
            scale=1/127.5
        )
    ],
    outputs=[
        ct.ImageType(
            name="Identity",  # Replace with actual output tensor name if different
            color_layout=ct.colorlayout.GRAYSCALE,
        )
    ]
)

# Optional: Print spec and save the model
spec = coreml_model.get_spec()

print(spec.description)

# Step 6: Save CoreML model
coreml_model.save("new_models/VERY_GOOD_CS_.mlpackage")
