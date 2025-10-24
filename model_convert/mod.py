import coremltools as ct
from coremltools.proto import Model_pb2

# Load model
mlmodel = ct.models.MLModel("new_models/image_as_input.mlpackage")
spec = mlmodel.get_spec()

# Get output feature
output = spec.description.output[0]

# Change from MultiArray to Image output
output.type.imageType.width = 256
output.type.imageType.height = 256
output.type.imageType.colorSpace = Model_pb2.ImageFeatureType.ColorSpace.Value("GRAYSCALE")

# Also, change the type to imageType
output.type.ClearField("multiArrayType")  # Remove old type
output.type.imageType.CopyFrom(Model_pb2.ImageFeatureType(
    width=256,
    height=256,
    colorSpace=Model_pb2.ImageFeatureType.ColorSpace.Value("GRAYSCALE")
))

# Save the updated model
updated_model = ct.models.MLModel(spec)
updated_model.save("new_models/image_as_input_output.mlpackage")