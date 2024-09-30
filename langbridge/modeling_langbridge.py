# code adapted from flamingo-mini https://github.com/dhansmair/flamingo-mini

from __future__ import annotations
from abc import ABC
from typing import Any, Dict, List
import contextlib
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import pandas as pd
from tqdm import tqdm

from transformers import PreTrainedModel, AutoModel, AutoConfig, PreTrainedTokenizer, MT5EncoderModel, UMT5EncoderModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast
)

from .configuration_langbridge import LangBridgeConfig
from .alignment_modules import LinearWithAddedEos, PerceiverResampler, FFNWithAddedEos, LinearWithRR, LinearWithAddedEosRR


@contextlib.contextmanager
def suppress_model_loading_warnings(suppress: bool = True):
    if suppress:
        logger = logging.getLogger('transformers.modeling_utils')
        level = logger.level
        logger.setLevel(logging.CRITICAL)
        yield
        logger.setLevel(level)
    else:
        yield


class LBBaseModel(ABC, PreTrainedModel):

    config: LangBridgeConfig
    enc: PreTrainedModel
    lm: PreTrainedModel
    lm_head: nn.Linear
    embeddings: nn.Embedding

    config_class = LangBridgeConfig

    def __init__(self, config: LangBridgeConfig, random_init=True, suppress_warnings=True):
        super().__init__(config)
        if 'umt5' in config.enc.lower():
            enc_class = UMT5EncoderModel
        elif 'mt5' in config.enc.lower():
            enc_class = MT5EncoderModel
        else:
            enc_class = AutoModel

        with suppress_model_loading_warnings(suppress_warnings):
            if random_init:
                enc_config = AutoConfig.from_pretrained(
                    config.enc)
                self.enc = enc_class(config=enc_config)
            else:
                print('loading encoder from pretrained')
                self.enc = enc_class.from_pretrained(config.enc)

        # self.enc.gradient_checkpointing_enable(
            # gradient_checkpointing_kwargs={'use_reentrant': False})

        if config.alignments == 'linear':  # default
            self.alignment = LinearWithAddedEos(
                dim=config.dim_enc, out_dim=config.dim_lm)
        elif config.alignments == 'ffn':  # mlp
            self.alignment = FFNWithAddedEos(
                dim=config.dim_enc, out_dim=config.dim_lm)
        elif config.alignments == 'latent':
            self.alignment = PerceiverResampler(
                dim=config.dim_enc, out_dim=config.dim_lm, num_latents=config.num_latents)
        elif config.alignments == 'rr':
            # # Initialize alignment as None so it will be set by `load_state_dict`
            # self.alignment = None
            assert config.anchor_ids is not None, 'anchor_ids must be provided for rr alignment'
            if isinstance(config.anchor_ids, dict):
                for key in config.anchor_ids:
                    if isinstance(config.anchor_ids[key], list):
                        # Convert each list back to a tensor
                        config.anchor_ids[key] = torch.tensor(config.anchor_ids[key])

            anchors = self._get_anchor_vectors(config.anchor_ids)
            self.alignment = LinearWithAddedEosRR(
                dim=config.dim_enc, out_dim=config.dim_lm, anchors=anchors)
        # elif config.alignments == 'rr':
        #     self.alignment = None
        else:
            raise ValueError(
                f'unknown alignment type {config.alignments}')

    def _get_anchor_vectors(self, anchor_ids: dict, batch_size: int = 32) -> Dict[str, torch.Tensor]:
        """Get the anchor vectors from the encoder model in batches."""

        assert 'high_res' in anchor_ids and 'low_res' in anchor_ids, 'anchor_ids must contain high_res and low_res keys'
        assert len(anchor_ids['high_res']) == len(anchor_ids['low_res']), 'high_res and low_res must have the same length'

        anchor_vectors = {}
        for res, anchors in tqdm(anchor_ids.items(), desc='Getting anchor vectors'):
            anchor_vectors[res] = []
            
            # Initialize inner progress bar for batching
            for i in tqdm(range(0, len(anchors), batch_size), desc=f'Getting anchor vectors for {res}'):
                batch_anchors = anchors[i:i + batch_size].to(self.enc.device)  # Shape: [batch_size, seq_length]
                
                with torch.no_grad():
                    model_output = self.enc(batch_anchors)
                    last_hidden_state = model_output.last_hidden_state  # Shape: [batch_size, seq_length, hidden_dim]

                # Compute the mean of the last hidden state across the sequence (dim=1)
                mean_hidden_state = last_hidden_state.mean(dim=1)  # Shape: [batch_size, hidden_dim]
                anchor_vectors[res].append(mean_hidden_state)

            # Concatenate all the batches to get a single tensor for each resolution
            anchor_vectors[res] = torch.cat(anchor_vectors[res], dim=0)  # Shape: [num_sentences, hidden_dim]

        return anchor_vectors

    # def initialize_alignment(self):
    #     # This method is called only during fresh initialization
    #     assert self.config.anchor_ids is not None, 'anchor_ids must be provided for rr alignment'
    #     anchors = get_anchor_vectors(self.config.anchor_ids, self.enc)
    #     self.alignment = LinearWithRR(
    #         dim=self.config.dim_enc, out_dim=self.config.dim_lm, anchors=anchors)

    # def load_state_dict(self, state_dict, strict=True):
    #     # If `alignment` was None, initialize it before loading the state_dict
    #     if self.config.alignments == 'rr' and self.alignment is None:
    #         self.initialize_alignment()
        
    #     # Load the state_dict normally
    #     super().load_state_dict(state_dict, strict)

    def freeze_encoder(self):
        """freeze vision model """
        for param in self.enc.parameters():
            param.requires_grad = False

    def freeze_lm(self):
        for param in self.lm.parameters():
            param.requires_grad = False

    def unfreeze_lm(self):
        for param in self.lm.parameters():
            param.requires_grad = True

    # get soft prompts
    def get_encoder_features(self, enc_ids: torch.Tensor, enc_mask: torch.Tensor) -> torch.Tensor:
        if self.config.freeze_encoder:
            with torch.no_grad():
                enc_features = self.enc(
                    input_ids=enc_ids, attention_mask=enc_mask).last_hidden_state  # (b, s, d)
        else:
            enc_features = self.enc(
                input_ids=enc_ids, attention_mask=enc_mask).last_hidden_state
        enc_features = self.alignment(enc_features, enc_mask)
        return enc_features

    def forward(
        self,
        enc_ids: torch.Tensor | None = None,
        enc_mask: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = True,
        past_key_values: tuple | None = None,
        return_dict: bool = True,
        labels: torch.Tensor | None = None,
        loss_reduction: str = 'mean',
        **kwargs
    ) -> CausalLMOutputWithPast:
        # sanity check
        assert return_dict, "can only use return_dict=True at the moment!"

        # find the input shape
        batch_size, seq_length = input_ids.shape[:
                                                 2] if input_ids is not None else enc_ids.shape[:2]
        device = input_ids.device if input_ids is not None else enc_ids.device
        lm_past_key_values = None if past_key_values is None else past_key_values[0]

        if input_ids is not None:
            embeddings = self.embeddings(input_ids)
        bos_shifted = False
        if lm_past_key_values is None:
            assert enc_ids.size(0) == batch_size

            enc_features = self.get_encoder_features(
                enc_ids, enc_mask)

            if input_ids is not None:
                first_input_ids = input_ids[:, 0]
                if all(first_input_ids == self.lm.config.bos_token_id):
                    # move bos embedding to the front
                    bos_shifted = True
                    embeddings = torch.cat(
                        [embeddings[:, 0].unsqueeze(dim=1), enc_features, embeddings[:, 1:]], dim=1)
                else:
                    embeddings = torch.cat([enc_features, embeddings], dim=1)
            else:
                embeddings = enc_features
            enc_feature_length = enc_features.shape[1]
        else:
            enc_feature_length = past_key_values[1]

        if input_ids is not None:
            if self.config.alignments not in ['linear', 'ffn', 'rr']:
                attn_mask = torch.cat(
                    [torch.ones((batch_size, enc_feature_length), device=device, dtype=torch.long), attention_mask], dim=1)
            else:
                # use the encoder masks for the soft prompts
                if bos_shifted:
                    # TODO: messy code
                    # torch.ones are there since the alignment adds a single learnable eos token at the enc
                    attn_mask = torch.cat(
                        [attention_mask[:, 0].unsqueeze(dim=1), enc_mask, torch.ones((batch_size, 1), device=device, dtype=torch.long), attention_mask[:, 1:]], dim=1)
                else:
                    attn_mask = torch.cat(
                        [enc_mask, torch.ones((batch_size, 1), device=device, dtype=torch.long), attention_mask], dim=1)
        else:
            attn_mask = enc_mask

        # if input_ids is not None:
        #     if self.config.alignments not in ['linear', 'ffn', 'rr']:
        #         # Ensure concatenation results in the correct length
        #         # print(f"Expected length: {enc_feature_length + attention_mask.shape[1]}")
        #         attn_mask = torch.cat(
        #             [torch.ones((batch_size, enc_feature_length), device=device, dtype=torch.long), attention_mask],
        #             dim=1
        #         )
        #     else:
        #         # Use encoder masks for soft prompts
        #         if bos_shifted:
        #             # Concatenate properly, ensuring the sequence lengths match
        #             # print(f"Expected length: {enc_mask.shape[1] + attention_mask.shape[1]}")
        #             attn_mask = torch.cat(
        #                 [enc_mask, attention_mask], dim=1
        #             )
        #         else:
        #             # Validate length to be correct
        #             # print(f"Expected length: {enc_mask.shape[1] + attention_mask.shape[1]}")
        #             attn_mask = torch.cat(
        #                 [enc_mask, attention_mask], dim=1
        #             )
        # else:
        #     attn_mask = enc_mask

        # expected_length = embeddings.shape[1]
        # if attention_mask.shape[1] < expected_length - enc_mask.shape[1]:
        #     padding = expected_length - enc_mask.shape[1] - attention_mask.shape[1]
        #     attention_mask = torch.cat([attention_mask, torch.ones((batch_size, padding), device=device, dtype=torch.long)], dim=1)

        # pass through LM
        out: BaseModelOutputWithPast = self.lm(
            attention_mask=attn_mask,
            inputs_embeds=embeddings,
            use_cache=use_cache,
            past_key_values=lm_past_key_values,
            return_dict=True,
            **kwargs
        )

        logits: torch.Tensor = self.lm_head(out.last_hidden_state)

        loss = None
        if labels is not None:
            # no loss for soft prompts
            no_loss_labels = torch.zeros(
                (batch_size, enc_feature_length), device=device, dtype=torch.long) + -100

            if bos_shifted:
                full_labels = torch.cat(
                    [labels[:, 0].unsqueeze(dim=1), no_loss_labels, labels[:, 1:]], dim=1)
            else:
                full_labels = torch.cat(
                    [no_loss_labels, labels], dim=1)
            # logits shape (batch, seq_length, #words)
            shift_logits = logits[..., :-1, :].contiguous()
            # labels shape (batch, seq_length)
            shift_labels = full_labels[..., 1:].contiguous()

            # Flatten the tokens
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                   shift_labels.view(-1), reduction=loss_reduction)
            if loss_reduction == 'none':
                # CrossEntropyLoss will flatten all dimensions by default
                loss = rearrange(loss, '(b s) -> b s', b=batch_size)
                # remove soft promtps from loss
                loss = loss[:, enc_feature_length:]
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=(out.past_key_values,
                             enc_feature_length) if use_cache else None,
            hidden_states=out.hidden_states,
            attentions=out.attentions,
        )


# used for debbuging with opt-125m
class LBOPT(LBBaseModel):
    config: LangBridgeConfig

    def __init__(self, config: LangBridgeConfig, random_init=True):
        from transformers import OPTForCausalLM, OPTModel
        super().__init__(config, random_init=random_init)

        if random_init:
            model_config = AutoConfig.from_pretrained(config.lm)
            base_lm: OPTForCausalLM = OPTForCausalLM(config=model_config)
        else:
            print('loading lm from pretrained')
            base_lm: OPTForCausalLM = OPTForCausalLM.from_pretrained(
                config.lm)
        assert self.config.dim_lm == base_lm.config.hidden_size, \
            f"specified {self.config.dim_lm=} in LangBridgeConfig, but {config.lm} has hidden size={base_lm.config.hidden_size}"

        self.lm: OPTModel = base_lm.model
        self.lm_head = base_lm.lm_head
        self.embeddings = base_lm.get_input_embeddings()

# used for Qwen model
class LBQwen(LBBaseModel):
    config: LangBridgeConfig

    def __init__(self, config: LangBridgeConfig, random_init=True):
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        import torch

        super().__init__(config, random_init=random_init)

        # Load model configuration and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.lm)

        # If random initialization is specified
        if random_init:
            model_config = AutoConfig.from_pretrained(config.lm)
            base_lm = AutoModelForCausalLM.from_config(model_config)
        else:
            # Load pretrained model
            print('Loading Qwen model from pretrained weights')
            base_lm = AutoModelForCausalLM.from_pretrained(
                config.lm,
                torch_dtype=torch.float16,  # Use the appropriate torch dtype
                device_map="auto"  # Load model onto the available device automatically
            )

        # Validate the hidden size matches the specified configuration
        assert self.config.dim_lm == base_lm.config.hidden_size, \
            f"specified {self.config.dim_lm=} in LangBridgeConfig, but {config.lm} has hidden size={base_lm.config.hidden_size}"

        # Extract components of the Qwen model
        self.lm = base_lm
        self.embeddings = base_lm.get_input_embeddings()

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # Perform forward pass through the model
        output = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        return output

    def generate_response(self, prompt, mode="CoT", max_new_tokens=512):
        # Define the system messages based on the mode
        if mode == "CoT":
            messages = [
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                {"role": "user", "content": prompt}
            ]
        elif mode == "TIR":
            messages = [
                {"role": "system", "content": "Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}."},
                {"role": "user", "content": prompt}
            ]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Apply the chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.lm.device)

        # Generate the response
        generated_ids = self.lm.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens
        )

        # Remove the input prompt tokens from the output
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response


class LBLlama(LBBaseModel):
    config: LangBridgeConfig

    def __init__(self, config: LangBridgeConfig, random_init=True):
        from transformers import LlamaForCausalLM, LlamaModel
        super().__init__(config, random_init=random_init)

        if random_init:
            model_config = AutoConfig.from_pretrained(config.lm)
            try:
                model_config.attn_implementation = 'flash_attention_2'
                base_lm: LlamaForCausalLM = LlamaForCausalLM(
                    config=model_config)
            except ImportError:
                print('Not using Flash Attention!')
                base_lm: LlamaForCausalLM = LlamaForCausalLM(
                    config=model_config)
        else:
            print('loading lm from pretrained')
            try:
                base_lm: LlamaForCausalLM = LlamaForCausalLM.from_pretrained(
                    config.lm, use_flash_attention_2=True)
            except ImportError:
                print('Not using Flash Attention!')
                base_lm: LlamaForCausalLM = LlamaForCausalLM.from_pretrained(
                    config.lm)

        assert self.config.dim_lm == base_lm.config.hidden_size, \
            f"specified {self.config.dim_lm=} in LangBridgeConfig, but {config.lm} has hidden size={base_lm.config.hidden_size}"

        self.lm: LlamaModel = base_lm.model
        self.lm_head = base_lm.lm_head
        self.embeddings = base_lm.get_input_embeddings()


class LBMistral(LBBaseModel):
    config: LangBridgeConfig

    def __init__(self, config: LangBridgeConfig, random_init=True):
        from transformers import MistralForCausalLM, MistralModel
        super().__init__(config, random_init=random_init)

        if random_init:
            model_config = AutoConfig.from_pretrained(config.lm)
            try:
                model_config.attn_implementation = 'flash_attention_2'
                base_lm: MistralForCausalLM = MistralForCausalLM(
                    config=model_config)
            except ImportError:
                print('Not using Flash Attention!')
                base_lm: MistralForCausalLM = MistralForCausalLM(
                    config=model_config)
        else:
            try:
                base_lm: MistralForCausalLM = MistralForCausalLM.from_pretrained(
                    config.lm, use_flash_attention_2=True)
            except ImportError:
                print('Not using Flash Attention!')
                base_lm: MistralForCausalLM = MistralForCausalLM.from_pretrained(
                    config.lm)
        assert self.config.dim_lm == base_lm.config.hidden_size, \
            f"specified {self.config.dim_lm=} in LangBridgeConfig, but {config.lm} has hidden size={base_lm.config.hidden_size}"

        self.lm: MistralModel = base_lm.model
        self.lm_head = base_lm.lm_head
        self.embeddings = base_lm.get_input_embeddings()


class LangBridgeModel(PreTrainedModel):
    config: LangBridgeConfig
    config_class = LangBridgeConfig

    _LANGUAGE_MODEL_VERSIONS = {
        'facebook/opt': LBOPT,
        'EleutherAI/llemma': LBLlama,
        'codellama/CodeLlama': LBLlama,
        'microsoft/Orca-2': LBLlama,
        'meta-math/MetaMath': LBLlama,
        'meta-llama/Llama-2-7b-hf': LBLlama,
        'meta-llama/Llama-3.2-1B-Instruct': LBLlama,
        'mistralai/Mistral-7B-v0.1': LBMistral,
        'Qwen/Qwen2.5': LBQwen,
    }

    def __init__(self, config: LangBridgeConfig, random_init=True, model_class=None):
        super().__init__(config)

        if model_class is None:
            model_class = self._find_lm_class(config.lm)
        self.lb: LBBaseModel = model_class(config, random_init=random_init)

        if config.freeze_language_model:
            self.freeze_lm()

        if config.freeze_encoder:
            self.freeze_encoder()

    @classmethod
    def _find_lm_class(cls, language_model_id: str):
        for prefix, lm_class in cls._LANGUAGE_MODEL_VERSIONS.items():
            if language_model_id.startswith(prefix):
                return lm_class
        raise ValueError(f'unsupported language model {language_model_id}')

    def freeze_encoder(self):
        self.lb.freeze_encoder()

    def freeze_lm(self):
        self.lb.freeze_lm()

    def unfreeze_lm(self):
        self.lb.unfreeze_lm()

    def forward(
        self,
        enc_ids: torch.Tensor | None = None,
        enc_mask: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = True,
        past_key_values: tuple | None = None,
        return_dict: bool = True,
        labels: torch.Tensor | None = None,
        loss_reduction: str = 'mean',
        **kwargs
    ) -> CausalLMOutputWithPast:

        return self.lb(
            enc_ids=enc_ids,
            enc_mask=enc_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            labels=labels,
            loss_reduction=loss_reduction,
            **kwargs
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        enc_ids: torch.Tensor | None = None,
        enc_mask: torch.Tensor | None = None,
        past=None,
        past_key_values=None,
        **kwargs
    ) -> Dict[str, Any]:
        """ hf specific function. Overridden from PreTrainedModel for text generation purposes.

        if use_cache is used, past is not None, then only the last column will be passed as input_ids.
        TODO was `past` renamed to `past_key_values` in transformers 4.26?
        """

        if past_key_values is not None or past is not None:
            input_ids = input_ids[:, -1:]

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            enc_ids=enc_ids,
            enc_mask=enc_mask,
            past_key_values=past_key_values if past_key_values is not None else past,
            **kwargs
        )

    def _reorder_cache(self, past, beam_idx):
        """ hf specific function. Overridden from PreTrainedModel.

        this is required for beam search in combination with use_cache.

        Args: 
            past is a tuple of past_key_values of the xattn layers, and of the LM layers.
            beam_idx: index of the beam
        """
        xattn_past, lm_past = past

        xattn_past_beam = tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device))
                  for past_state in layer_past)
            for layer_past in xattn_past
        )

        lm_past_beam = tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device))
                  for past_state in layer_past)
            for layer_past in lm_past
        )

        return xattn_past_beam, lm_past_beam

    # a simple function to test the model
    @torch.no_grad()
    def generate_from_prefix(
        self,
        enc_tokenizer: PreTrainedTokenizer,
        lm_tokenizer: PreTrainedTokenizer,
        prompts: List[str],
        **kwargs
    ):
        enc_input = enc_tokenizer(prompts, return_tensors='pt', padding=True)
        enc_ids = enc_input['input_ids'].to(self.device)
        enc_mask = enc_input['attention_mask'].to(self.device)

        input_ids = torch.LongTensor([lm_tokenizer.bos_token_id])
        input_ids = input_ids.repeat(enc_ids.shape[0], 1).to(self.device)
        attention_mask = torch.ones_like(input_ids)

        # print(lm_tokenizer.eos_token_id)

        # EOS issue fixed here for normal usage
        out_ids = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            enc_ids=enc_ids,
            enc_mask=enc_mask,
            early_stopping=True,
            use_cache=True,
            bos_token_id=lm_tokenizer.bos_token_id,
            # eos_token_id=32002,  # <|im_end|>
            eos_token_id=lm_tokenizer.eos_token_id,
            pad_token_id=lm_tokenizer.eos_token_id,
            **kwargs
        )

        completions = lm_tokenizer.batch_decode(
            out_ids, skip_special_tokens=True)
        # TODO: don't know why batch_decode doesn't remove <|im_end|>, since it's in the special tokens
        completions = [s.replace('<|im_end|>', '') for s in completions]
        return completions
