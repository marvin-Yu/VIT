import torch
import sys
import fused_bert
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import (
    BaseModelOutput)

# BertEncoder targets that we can accelerate
hook_targets = [
    'transformers.models.roberta.modeling_roberta.RobertaEncoder',
    'transformers.models.bert.modeling_bert.BertEncoder',
]

def fullname(o):
    module = o.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return o.__class__.__name__
    return module + '.' + o.__class__.__name__

def find_target_module(model):
    for key, sub_model in model._modules.items():
        name = fullname(sub_model)

        if name in hook_targets:
            return sub_model

        r = find_target_module(sub_model)
        if r:
            return r

    return None

def optimize_bert_encoder(model, is_int8=False):
    fused_bert.init({
        "num_hidden_layers": model.config.num_hidden_layers,
        "hidden_size": model.config.hidden_size,
        "intermediate_size": model.config.intermediate_size,
        "num_attention_heads": model.config.num_attention_heads,
        "is_int8": 1 if is_int8 else 0
    })

    # Load weights
    weights = model.state_dict()
    weight_names = weights.keys()
    encoder_prefix = None
    target_str = '.layer.0.attention.attention.query.weight'
    for name in weight_names:
        if name.endswith(target_str):
            encoder_prefix = name[:-len(target_str)]
            break

    if not encoder_prefix:
        print("Canot find the weight for bert encoder.")
        sys.exit(-1)

    fused_bert.init_weights(weights, encoder_prefix)

    # Replace VitEncoder's forward method with bert_encoder_forward
    # The forward method of BertEncoder is in modeling_bert.py
    def forward(
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        result = fused_bert.forward(hidden_states)
        return BaseModelOutput(
        last_hidden_state=result,
        hidden_states=None,
        attentions=None,
    )

    # Try to detect the BertEncoder
    succeed = False
    try:
        model.encoder.forward = forward
        succeed = True
    except:
        encoder = find_target_module(model)
        if encoder:
            encoder.forward = forward
            succeed = True

    if not succeed:
        print("Canot find bert encoder in the model.")
        sys.exit(-1)
