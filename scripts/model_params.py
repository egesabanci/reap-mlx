import sys
import torch
from transformers import AutoModelForCausalLM
from reap.model_util import MODEL_ATTRS, patched_model_map

model_name = sys.argv[1]
patched_model_name = patched_model_map(model_name)
model = AutoModelForCausalLM.from_pretrained(
    patched_model_name, 
    device_map="auto", 
    torch_dtype="auto", 
    trust_remote_code=True
)


def get_total_params(model, compression_ratio):
    """Calculates the total number of parameters in the model with compression on experts."""
    model_attrs = MODEL_ATTRS.get(model.__class__.__name__)
    if not model_attrs:
        raise ValueError(f"Model {model.__class__.__name__} not supported.")

    expert_param_names = set()
    moe_block_name = model_attrs['moe_block']
    
    for i in range(len(model.model.layers)):
        # Check if the layer has a MoE block
        if hasattr(model.model.layers[i], moe_block_name):
            moe_block = getattr(model.model.layers[i], moe_block_name)
            if hasattr(moe_block, model_attrs["experts"]):
                experts = getattr(moe_block, model_attrs["experts"])
                for name, _ in experts.named_parameters():
                    full_name = f"model.layers.{i}.{moe_block_name}.experts.{name}"
                    expert_param_names.add(full_name)

    total_params = 0
    expert_params = 0

    for name, param in model.named_parameters():
        if name in expert_param_names:
            expert_params += param.numel()
        else:
            total_params += param.numel()
            
    compressed_total_params = total_params + (expert_params * (1-compression_ratio))
    return compressed_total_params

for compression_ratio in [0.0, 0.25, 0.5]:
    print(f"Calculating total parameters with {compression_ratio*100}% expert compression...")
    total_params = get_total_params(model, compression_ratio)
    print(f"Total parameters with {compression_ratio*100}% expert compression: {total_params}")