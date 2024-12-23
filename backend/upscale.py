import os
from mmagic.apis import MMagicInferencer
from mmengine import mkdir_or_exist

# Define input and output paths
video = './output/cropped_number_plate.mp4'
result_out_dir = './output/cropped_number_plate_upscaled.mp4'

# Initialize MMagicInferencer with the BasicVSR model
editor = MMagicInferencer('basicvsr')

# Perform inference and save the result
results = editor.infer(video=video, result_out_dir=result_out_dir)

# Optionally, you can print the results or inspect any other outputs
print("Upscaled video saved to:", result_out_dir)