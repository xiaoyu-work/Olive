# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def convert_kohya_lora_to_diffusers(state_dict):
    unet_state_dict = {}
    te_state_dict = {}
    network_alpha = None

    for key, value in state_dict.items():
        if "lora_down" in key:
            lora_name = key.split(".")[0]
            lora_name_up = lora_name + ".lora_up.weight"
            lora_name_alpha = lora_name + ".alpha"
            if lora_name_alpha in state_dict:
                alpha = state_dict[lora_name_alpha].item()
                if network_alpha is None:
                    network_alpha = alpha
                elif network_alpha != alpha:
                    raise ValueError("Network alpha is not consistent")

            if lora_name.startswith("lora_unet_"):
                diffusers_name = key.replace("lora_unet_", "").replace("_", ".")
                diffusers_name = diffusers_name.replace("down.blocks", "down_blocks")
                diffusers_name = diffusers_name.replace("mid.block", "mid_block")
                diffusers_name = diffusers_name.replace("up.blocks", "up_blocks")
                diffusers_name = diffusers_name.replace("transformer.blocks", "transformer_blocks")
                diffusers_name = diffusers_name.replace("to.q.lora", "to_q_lora")
                diffusers_name = diffusers_name.replace("to.k.lora", "to_k_lora")
                diffusers_name = diffusers_name.replace("to.v.lora", "to_v_lora")
                diffusers_name = diffusers_name.replace("to.out.0.lora", "to_out_lora")
                if "transformer_blocks" in diffusers_name:
                    if "attn1" in diffusers_name or "attn2" in diffusers_name:
                        diffusers_name = diffusers_name.replace("attn1", "attn1.processor")
                        diffusers_name = diffusers_name.replace("attn2", "attn2.processor")
                        unet_state_dict[diffusers_name] = value
                        unet_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict[lora_name_up]
            elif lora_name.startswith("lora_te_"):
                diffusers_name = key.replace("lora_te_", "").replace("_", ".")
                diffusers_name = diffusers_name.replace("text.model", "text_model")
                diffusers_name = diffusers_name.replace("self.attn", "self_attn")
                diffusers_name = diffusers_name.replace("q.proj.lora", "to_q_lora")
                diffusers_name = diffusers_name.replace("k.proj.lora", "to_k_lora")
                diffusers_name = diffusers_name.replace("v.proj.lora", "to_v_lora")
                diffusers_name = diffusers_name.replace("out.proj.lora", "to_out_lora")
                if "self_attn" in diffusers_name:
                    te_state_dict[diffusers_name] = value
                    te_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict[lora_name_up]

    unet_state_dict = {f"unet.{module_name}": params for module_name, params in unet_state_dict.items()}
    te_state_dict = {f"text_encoder.{module_name}": params for module_name, params in te_state_dict.items()}
    new_state_dict = {**unet_state_dict, **te_state_dict}
    return new_state_dict, network_alpha
