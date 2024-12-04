import torch
import os
from zero_to_16bit import get_fp32_state_dict_from_zero_checkpoint

model_state_dict = get_fp32_state_dict_from_zero_checkpoint('/Train_Result/VLM-PT-32B/checkpoint-100', None)
model_state_dict = {k: v.bfloat16() for k, v in model_state_dict.items()}

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                pass
                #logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k.replace('mlp1.', ''): maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k.replace('base_model.model.', ''): maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return

output_dir = './Sava_Model/VLM-SFT-lora/'

non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model_state_dict.items())
state_dict = get_peft_state_maybe_zero_3(model_state_dict.items(), bias="none")

torch.save(non_lora_state_dict, os.path.join(output_dir, 'non_lora_trainables.bin'))
torch.save(state_dict, os.path.join(output_dir, 'adapter_model.bin')) 
                                                        
                                                        
                                                      
