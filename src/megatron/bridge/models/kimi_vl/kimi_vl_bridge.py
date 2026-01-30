

from typing import Dict, Mapping

import torch
import torch.nn as nn
from transformers.generation.utils import GenerationMixin
from transformers.modeling_utils import PreTrainedModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, WeightConversionTask
from megatron.bridge.models.deepseek.common import get_common_configs
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM
from megatron.bridge.models.kimi_vl.kimi_vl_provider import KimiVLMoEModelProvider
from megatron.bridge.models.kimi_vl.modelling_kimi_vl.model import KimiVLModel
from megatron.bridge.models.kimi_vl.modelling_kimi_vl.transfomer_config import KimiVLConfig


class KimiVLPreTrainedModel(PreTrainedModel, GenerationMixin):
    config_class = KimiVLConfig
    base_model_prefix = "model"
    _no_split_modules = ["MoonVitEncoderLayer", "DeepseekV3DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = False

    def _init_weights(self, module):
        # important: this ported version of Llava isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed - the original codebase
        # https://github.com/haotian-liu/LLaVA/tree/main/llava should serve for that purpose
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class KimiVLForConditionalGeneration(KimiVLPreTrainedModel, GenerationMixin):
    def __init__(self, config: KimiVLConfig):
        super().__init__(config)


@MegatronModelBridge.register_bridge(source=KimiVLForConditionalGeneration, target=KimiVLModel)
class KimiVLMoEBridge(MegatronModelBridge):

    def provider_bridge(self, hf_pretrained: PreTrainedVLM) -> KimiVLMoEModelProvider:       
        hf_config = hf_pretrained.config
        text_config = hf_config.text_config

        model_dtype = self.dtype_from_hf(hf_config, default=torch.float32)
        
        vision_config = hf_config.vision_config
        vision_config.torch_dtype = model_dtype
        
        breakpoint()
        

        provider = KimiVLMoEModelProvider(
            # Language model configuration from text_config
            num_layers=text_config.num_hidden_layers,
            hidden_size=text_config.hidden_size,
            ffn_hidden_size=text_config.intermediate_size,  # Dense FFN size (for non-MoE layers if any)
            moe_ffn_hidden_size=text_config.moe_intermediate_size,  # Expert FFN size
            num_attention_heads=text_config.num_attention_heads,
            init_method_std=text_config.initializer_range,
            layernorm_epsilon=text_config.rms_norm_eps,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(text_config.vocab_size),
            rotary_base=text_config.rope_theta,
            share_embeddings_and_output_weights=getattr(text_config, "tie_word_embeddings", False),
            vocab_size=text_config.vocab_size,
            seq_length=text_config.max_position_embeddings,
            fp16=(model_dtype == torch.float16),
            bf16=(model_dtype == torch.bfloat16),
            params_dtype=model_dtype,
            generation_config=hf_pretrained.generation_config,
            vision_config=vision_config,
            hf_text_config=text_config,
        )

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        """
        Return MegatronMappingRegistry containing parameter mappings for MoE models.

        The MoE mappings include:
        1. Standard language model mappings (embeddings, layer norms, output)
        2. Vision model mappings (same as dense model)
        3. QKV mappings with QK layernorm
        4. MoE-specific mappings:
           - Router weights for expert selection
           - Expert MLPs (multiple experts per layer)
           - Pre-MLP layernorm
        5. Deepstack visual merger mappings

        Returns:
            MegatronMappingRegistry with all MoE parameter mappings
        """
        # Language model direct mappings (same as dense model)
        param_mappings = {
            # Embeddings and output layers
            "language_model.embedding.word_embeddings.weight": "model.language_model.embed_tokens.weight",
            "language_model.output_layer.weight": "lm_head.weight",
            "language_model.decoder.final_layernorm.weight": "model.language_model.norm.weight",
            # Layer normalization for attention
            "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.language_model.layers.*.input_layernorm.weight",
            # MoE-specific: pre-MLP layernorm
            "language_model.decoder.layers.*.pre_mlp_layernorm.weight": "model.language_model.layers.*.post_attention_layernorm.weight",
            # Attention output projection
            "language_model.decoder.layers.*.self_attention.linear_proj.weight": "model.language_model.layers.*.self_attn.o_proj.weight",
            # QK layernorm weights (Qwen3 specific)
            "language_model.decoder.layers.*.self_attention.q_layernorm.weight": "model.language_model.layers.*.self_attn.q_norm.weight",
            "language_model.decoder.layers.*.self_attention.k_layernorm.weight": "model.language_model.layers.*.self_attn.k_norm.weight",
            # MoE router weights
            "language_model.decoder.layers.*.mlp.router.weight": "model.language_model.layers.*.mlp.gate.weight",
        }

        mapping_list = []

        # Convert simple 1:1 mappings to AutoMapping objects
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # Add special mappings that require parameter transformation
        mapping_list.extend(
            [
                # Vision model weights are replicated directly
                ReplicatedMapping(
                    megatron_param="vision_model.**",
                    hf_param="model.visual.**",
                ),
                # QKV mapping: Combine separate Q, K, V matrices
                QKVMapping(
                    megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.language_model.layers.*.self_attn.q_proj.weight",
                    k="model.language_model.layers.*.self_attn.k_proj.weight",
                    v="model.language_model.layers.*.self_attn.v_proj.weight",
                ),
                # QKV bias mapping (if attention_bias is True)
                QKVMapping(
                    megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.bias",
                    q="model.language_model.layers.*.self_attn.q_proj.bias",
                    k="model.language_model.layers.*.self_attn.k_proj.bias",
                    v="model.language_model.layers.*.self_attn.v_proj.bias",
                ),
                ExpertMLPGateUpProjMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.experts.linear_fc1.weight*",
                    hf_param="model.language_model.layers.*.mlp.experts.gate_up_proj",
                ),
                ExpertMLPDownProjMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.experts.linear_fc2.weight*",
                    hf_param="model.language_model.layers.*.mlp.experts.down_proj",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)

    def maybe_modify_converted_hf_weight(
        self,
        task: WeightConversionTask,
        converted_weights_dict: Dict[str, torch.Tensor],
        hf_state_dict: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        num_experts = self.hf_config.text_config.num_experts
        ep_size = parallel_state.get_expert_model_parallel_world_size()
        experts_per_rank = num_experts // ep_size

        try:
            local_expert_number = extract_expert_number_from_param(task.param_name) % experts_per_rank
        except ValueError:
            # not an expert weight
            return converted_weights_dict

        assert len(converted_weights_dict) == 1, (
            f"There should be only one key in the converted_weights_dict, got keys: {converted_weights_dict.keys()}"
        )
        for key, value in converted_weights_dict.items():
            if key not in self.hf_weights_cache:
                self.hf_weights_cache[key] = {}

            # we end up with ep_size many weights to add to the cache
            # unpack the weights and re-index
            if ep_size == 1:
                self.hf_weights_cache[key][local_expert_number] = value
            else:
                assert value.shape[0] == ep_size
                for i, exp_val in enumerate(value):
                    global_expert_number = local_expert_number + (i * experts_per_rank)
                    self.hf_weights_cache[key][global_expert_number] = exp_val
            if len(self.hf_weights_cache[key]) == num_experts:
                logging.debug(f"All experts are loaded for {key}")
                # all experts are loaded
                if self.hf_weights_cache[key][0].ndim == 3:  # expert 0
                    # gate up
                    merged_hf_gate_weights = torch.cat(
                        [self.hf_weights_cache[key][i][0].unsqueeze(0) for i in range(num_experts)], dim=0
                    )
                    merged_hf_up_weights = torch.cat(
                        [self.hf_weights_cache[key][i][1].unsqueeze(0) for i in range(num_experts)], dim=0
                    )
                    del self.hf_weights_cache[key]
                    return {key: torch.cat([merged_hf_gate_weights, merged_hf_up_weights], dim=-1)}
                elif self.hf_weights_cache[key][0].ndim == 2:  # expert 0
                    # down
                    merged_hf_down_weights = torch.cat(
                        [self.hf_weights_cache[key][i].unsqueeze(0) for i in range(num_experts)], dim=0
                    )
                    del self.hf_weights_cache[key]
                    return {key: merged_hf_down_weights}
                else:
                    raise ValueError(
                        f"Incorrect shape of self.hf_weights_cache[key]: {key} {self.hf_weights_cache[key].shape}"
                    )
            else:
                # not all experts are loaded yet, return empty dict
                logging.debug(f"{len(self.hf_weights_cache[key])}/{num_experts} experts are loaded for {key}")
                return {}