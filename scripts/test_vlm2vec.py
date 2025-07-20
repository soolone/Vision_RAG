import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'VLM2Vec'))

from src.arguments import ModelArguments, DataArguments
from src.model.model import MMEBModel
from src.model.processor import load_processor, QWEN2_VL, VLM_IMAGE_TOKENS, Qwen2_VL_process_fn
from src.utils import batch_to_device
from PIL import Image
import torch

model_args = ModelArguments(
    model_name='Qwen/Qwen2-VL-7B-Instruct',
    checkpoint_path='TIGER-Lab/VLM2Vec-Qwen2VL-7B',
    pooling='last',
    normalize=True,
    model_backbone='qwen2_vl',
    lora=True
)
data_args = DataArguments()

processor = load_processor(model_args, data_args)
model = MMEBModel.load(model_args)
model = model.to('cuda', dtype=torch.bfloat16)
model.eval()

# Image + Text -> Text
inputs = processor(text=f'{VLM_IMAGE_TOKENS[QWEN2_VL]} Represent the given image with the following question: What is in the image',
                   images=Image.open('data/car.jpg'),
                   return_tensors="pt")
inputs = {key: value.to('cuda') for key, value in inputs.items()}
inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
inputs['image_grid_thw'] = inputs['image_grid_thw'].unsqueeze(0)
outputs = model(qry=inputs)
qry_output = outputs["qry_reps"]
# Text -> Text
string = 'A woman leaning against the car.'
inputs = processor(text=string,
                   images=None,
                   return_tensors="pt")
inputs = {key: value.to('cuda') for key, value in inputs.items()}
tgt_output = model(tgt=inputs)["tgt_reps"]
print(string, '=', model.compute_similarity(qry_output, tgt_output))