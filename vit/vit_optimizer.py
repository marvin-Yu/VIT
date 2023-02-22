import torch
import sys
import fused_bert
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions)

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

    # Replace BertEncoder's forward method with bert_encoder_forward
    # The forward method of BertEncoder is in modeling_bert.py
    def fused_forward_1_12(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        # print("marvin:", type(attention_mask), attention_mask.shape)
        attention_mask = torch.zeros(size=(1, 1, 1, 32))
        result = fused_bert.forward(hidden_states, attention_mask)
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=result, past_key_values=None, hidden_states=None, attentions=None, cross_attentions=None)

    # Try to detect the BertEncoder
    succeed = False
    try:
        model.encoder.forward = fused_forward_1_12
        succeed = True
    except:
        encoder = find_target_module(model)
        if encoder:
            encoder.forward = fused_forward_1_12
            succeed = True

    if not succeed:
        print("Canot find bert encoder in the model.")
        sys.exit(-1)
