from ._base import BaseGPTQForCausalLM
from ._base_Latte import BaseGPTQForLatte

class WrappedLatte(BaseGPTQForLatte):
    layer_type = "LatteT2V"
    layers_block_name = "transformer_blocks"
    temporal_layers_block_name = "temporal_transformer_blocks"
    outside_layer_modules = [
        "pos_embed",
        "adaln_single",
        "proj_out",
        "caption_projection"
    ]
    inside_layer_modules = [
        ["attn1.to_q", "attn1.to_k", "attn1.to_v"],
        ["attn1.to_out.0"],
        ["attn2.to_q", "attn2.to_k", "attn2.to_v"],
        ["attn2.to_out.0"],
        ["ff.net.0.proj"],
        ["ff.net.2"],
    ]


__all__ = ["WrappedLatte"]
