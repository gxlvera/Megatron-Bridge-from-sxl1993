from typing import Optional

import torch
import transformers
from megatron.core.models.gpt import GPTModel
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import divide
from packaging.version import Version as PkgVersion
from torch import nn

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.kimi_vl.modelling_kimi_vl.moonvit import MoonVitPretrainedModel
from megatron.bridge.models.kimi_vl.modelling_kimi_vl.transfomer_config import (
    KiMi25VLTransformerConfig,
    KimiVLConfig,
    KimiVLMultimodalProjectorConfig,
    MoonViTConfig,
)
from megatron.bridge.utils.common_utils import hook_hf_module_setattr_for_tp_grad_sync


def is_transformers_min_version(version):
    """Check if minimum version of transformers is installed."""
    try:
        transformers_version = PkgVersion(transformers.__version__)
        return transformers_version >= PkgVersion(version)
    except Exception:
        # If version parsing fails, assume false for safety
        return False


class KimiVLModel(MegatronModule):
    def __init__(
        self,
        config: GPTModelProvider,
        vision_transformer_config: MoonViTConfig,
        multi_modal_projector_config: KimiVLMultimodalProjectorConfig,
        pre_process: bool = True,
        post_process: bool = True,
        vp_stage: Optional[int] = None,
    ) -> None:
        super().__init__(config=config)

        self.pre_process = pre_process
        self.post_process = post_process
        self.vp_stage = vp_stage

        self.vision_model = None
        # self.image_token_id = language_transformer_config.image_token_id
        # self.video_token_id = language_transformer_config.video_token_id
        # self.vision_start_token_id = language_transformer_config.vision_start_token_id

        # This attribute is needed to check if an all-reduce is required
        # on the word embeddings inside `finalize_model_grads._allreduce_word_embedding_grads`.
        self.share_embeddings_and_output_weights = False

        if self.pre_process:
            # Initialize vision model with random weights from config
            self.vision_model = MoonVitPretrainedModel._from_config(vision_transformer_config)
            self.multi_modal_projector = KimiVLMultiModalProjector(multi_modal_projector_config)

            # Ensure HF visual tower params are marked for TP grad sync and future assignments are hooked.
            hook_hf_module_setattr_for_tp_grad_sync(self.vision_model)

        self.language_model = self.config.provide_language_model(
            pre_process=pre_process, post_process=post_process, vp_stage=vp_stage
        )

    # def shared_embedding_or_output_weight(self):
    #     """This is a convenience method to surface the language model's word embeddings, which is
    #     necessary for `finalize_model_grads._allreduce_word_embedding_grads`."""
    #     if self.add_decoder:
    #         return self.language_model.shared_embedding_or_output_weight()
    #     return None

    # def set_input_tensor(self, input_tensor) -> None:
    #     # This is usually handled in schedules.py but some inference code still
    #     # gives us non-lists or None
    #     if not isinstance(input_tensor, list):
    #         input_tensor = [input_tensor]
    #     assert len(input_tensor) == 1, "input_tensor should only be length 1 for Qwen3VL"

    #     if self.pre_process:
    #         self.encoder_hidden_state = input_tensor[0]
    #     else:
    #         self.language_model.set_input_tensor(input_tensor[0])

    def freeze(
        self,
        freeze_language_model: bool,
        freeze_vision_model: bool,
        freeze_vision_projection: bool,
    ):
        modules = []

        if freeze_language_model and self.language_model is not None:
            modules.append(self.language_model)

        if freeze_vision_model and self.vision_model is not None:
            # Freeze vision encoder components (patch_embed, blocks, pos_embed, rotary_pos_emb)
            if hasattr(self.vision_model, "patch_embed"):
                modules.append(self.vision_model.patch_embed)
            if hasattr(self.vision_model, "encoder"):
                modules.append(self.vision_model.encoder)

        if freeze_vision_projection and self.vision_model is not None:
            # Freeze vision projection components (merger and deepstack_merger_list)
            if hasattr(self.multi_modal_projector, "pre_norm"):
                modules.append(self.multi_modal_projector.pre_norm)
            if hasattr(self.multi_modal_projector, "linear_1"):
                modules.append(self.multi_modal_projector.linear_1)
            if hasattr(self.multi_modal_projector, "act"):
                modules.append(self.multi_modal_projector.act)
            if hasattr(self.multi_modal_projector, "linear_2"):
                modules.append(self.multi_modal_projector.linear_2)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    # def forward(
    #     self,
    #     input_ids: torch.Tensor,
    #     position_ids: torch.Tensor = None,  # can set at dataset
    #     attention_mask: torch.Tensor = None,
    #     labels: torch.Tensor = None,
    #     loss_mask: torch.Tensor = None,
    #     inference_params: InferenceParams = None,
    #     packed_seq_params: PackedSeqParams = None,
    #     extra_block_kwargs: dict = None,
    #     pixel_values: torch.Tensor = None,
    #     pixel_values_videos: torch.Tensor = None,
    #     image_grid_thw: torch.Tensor = None,
    #     video_grid_thw: torch.Tensor = None,
    #     # cat set at dataset
    #     image_input_mask: torch.Tensor = None,
    # ) -> torch.Tensor:
    #     """Forward function of the Qwen3VL model.

    #     Args:
    #         image_data (torch.Tensor): input image of shape [total_thw_size, n_features].
    #         input_ids (torch.Tensor): input text ids [batch, text_seq_len].
    #         position_ids (torch.Tensor): input text position ids [batch, text_seq_len].
    #         attention_mask (torch.Tensor): attention mask for the language model [batch, 1, combined_seq_len,
    #             combined_seq_len].
    #         labels (torch.Tensor): Optional target text labels [batch, combined_seq_len].
    #         inference_params (InferenceParams): Inference-time parameters including KV cache.

    #         video_start_index:
    #             0 -- all video
    #             len(video_seq) -- all image
    #             others -- mixture
    #         *_input_mask: should not be None in the first PP stage
    #     Returns:
    #         output (torch.Tensor): Loss of shape [b, s] if labels are provided, otherwise logits of shape
    #             [b, s, vocab_size].
    #     """
    #     assert pixel_values_videos is None and video_grid_thw is None, "not support video now"
    #     assert inference_params is None, "not support inference"
    #     breakpoint()
    #     video_start_index = 0
    #     vision_grid_thw = None
    #     vision_data = None
    #     image_mask = None
    #     deepstack_feature_lists = None
    #     # position ids is computed within the model
    #     position_ids = None

    #     if self.pre_process:
    #         if image_grid_thw is not None:
    #             image_mask = image_input_mask
    #             if image_mask is None:
    #                 image_mask = (input_ids == self.image_token_id).contiguous()
    #             vision_grid_thw = image_grid_thw
    #             vision_data = pixel_values
    #             video_start_index = image_mask.sum().item()
    #             assert video_start_index > 0

    #         vision_embeds = None
    #         if vision_grid_thw is not None and vision_grid_thw.shape[0] > 0:
    #             vision_embeds, deepstack_feature_lists = self.vision_model(
    #                 hidden_states=vision_data,
    #                 grid_thw=vision_grid_thw,
    #             )

    #         combined_embeddings = self.language_model.embedding(
    #             input_ids=input_ids,
    #             position_ids=None,  # NOTE: disable
    #         ).clone()  # [text_seq_len, b, h_language]

    #         if vision_embeds is not None:
    #             if video_start_index == 0:
    #                 image_embeds = None
    #                 video_embeds = vision_embeds
    #             elif video_start_index == vision_embeds.shape[0]:
    #                 image_embeds = vision_embeds
    #                 video_embeds = None
    #             elif 0 < video_start_index < vision_embeds.shape[0]:
    #                 image_embeds = vision_embeds[:video_start_index]
    #                 video_embeds = vision_embeds[video_start_index:]
    #             else:
    #                 raise ValueError(
    #                     f"Expect video token start index in range [0, {vision_embeds.shape[0]}], but got "
    #                     f"{video_start_index}"
    #                 )
    #             assert video_embeds is None, "not support video now"

    #             if image_embeds is not None:
    #                 combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()
    #                 combined_embeddings[image_mask] = image_embeds
    #                 combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()
    #         if self.config.sequence_parallel:
    #             combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(combined_embeddings)
    #             combined_embeddings = combined_embeddings.contiguous()
    #     else:
    #         combined_embeddings = None
    #     cu_seqlens_padded = None
    #     if packed_seq_params is not None:
    #         if packed_seq_params.cu_seqlens_q_padded is not None:
    #             cu_seqlens_padded = packed_seq_params.cu_seqlens_q_padded
    #         else:
    #             cu_seqlens_padded = packed_seq_params.cu_seqlens_q
    #     if position_ids is None:
    #         input_ids_for_rope_index = input_ids
    #         if cu_seqlens_padded is not None:
    #             def thd_to_bshd(packed_values: torch.Tensor, cu_seqlens: torch.Tensor):
    #                 seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    #                 max_seq_len = seqlens.max()
    #                 bs = len(cu_seqlens) - 1
    #                 results = packed_values.new_zeros(size=(bs, max_seq_len, *packed_values.shape[2:]))
    #                 for i, seqlen in enumerate(seqlens):
    #                     results[i, :seqlen] = packed_values[0, cu_seqlens[i]: cu_seqlens[i] + seqlen]
    #                 return results

    #             def bshd_to_thd(unpacked_values: torch.Tensor, cu_seqlens: torch.Tensor):
    #                 seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    #                 total_len = cu_seqlens[-1]
    #                 results = unpacked_values.new_zeros(size=(1, total_len, *unpacked_values.shape[2:]))
    #                 for i, seqlen in enumerate(seqlens):
    #                     results[0, cu_seqlens[i]: cu_seqlens[i] + seqlen] = unpacked_values[i, :seqlen]
    #                 return results

    #             input_ids_for_rope_index = thd_to_bshd(input_ids, cu_seqlens_padded)

    #         position_ids, _ = get_rope_index(
    #             self.config.spatial_merge_size,
    #             self.image_token_id,
    #             self.video_token_id,
    #             self.vision_start_token_id,
    #             input_ids_for_rope_index,
    #             image_grid_thw=image_grid_thw,
    #             video_grid_thw=video_grid_thw,
    #             attention_mask=attention_mask,
    #             packed_seq_params=packed_seq_params,
    #         )
    #         if cu_seqlens_padded is not None:
    #             position_ids = bshd_to_thd(position_ids.permute(1, 2, 0), cu_seqlens_padded).permute(2, 0, 1)

    #     visual_pos_masks = image_mask
    #     deepstack_visual_embeds = deepstack_feature_lists
    #     if self.config.sequence_parallel:
    #         visual_pos_masks, deepstack_visual_embeds = split_deepstack_embs(
    #             visual_pos_masks,
    #             deepstack_visual_embeds,
    #             tp_size=mpu.get_tensor_model_parallel_world_size(),
    #             tp_rank=mpu.get_tensor_model_parallel_rank(),
    #         )

    #     output = self.language_model(
    #         input_ids=None,
    #         position_ids=position_ids,  # None in encoder
    #         attention_mask=attention_mask,  # None in encoder
    #         decoder_input=combined_embeddings,  # only not None in the first decoder PP stage
    #         labels=labels,  # only not None in the last decoder PP stage
    #         loss_mask=loss_mask,
    #         inference_params=inference_params,  # currently always None
    #         packed_seq_params=packed_seq_params,  # currently always None
    #         visual_pos_masks=visual_pos_masks,
    #         deepstack_visual_embeds=deepstack_visual_embeds,
    #         **(extra_block_kwargs or {}),
    #     )

    #     return output


class KimiVLMultiModalProjector(MegatronModule):
    """Megatron-style MultiModal Projector for Vision→Text alignment"""

    def __init__(self, config: TransformerConfig):
        super().__init__(config=config)

        vision_hidden_size = config.input_size
        merge_k = config.merge_kernel_size
        self.hidden_size = vision_hidden_size * merge_k[0] * merge_k[1]

        # LayerNorm before projection
        self.pre_norm = nn.LayerNorm(vision_hidden_size, eps=config.layernorm_epsilon)

        # Linear 1 — Column Parallel (split output dimension across GPUs)
        self.linear_1 = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,
            init_method=config.init_method,
            bias=True,
            gather_output=False,
            skip_bias_add=False,
            config=config,
        )

        # Activation
        self.act = nn.GELU()

        # Linear 2 — Row Parallel (split input dimension)
        self.linear_2 = RowParallelLinear(
            self.hidden_size,
            config.hidden_size,
            init_method=config.init_method,
            bias=True,
            input_is_parallel=False,
            skip_bias_add=False,
            config=config,
        )

    def forward(self, image_features: list[torch.Tensor]) -> torch.Tensor:
        # Gather image features
        image_features = torch.cat(image_features, dim=0)

        # LayerNorm + reshape
        hidden_states = self.pre_norm(image_features).view(-1, self.hidden_size)

        # First Linear projection
        hidden_states, _ = self.linear_1(hidden_states)

        # Activation
        hidden_states = self.act(hidden_states)

        # Second projection to text hidden size
        hidden_states, _ = self.linear_2(hidden_states)

        return hidden_states
