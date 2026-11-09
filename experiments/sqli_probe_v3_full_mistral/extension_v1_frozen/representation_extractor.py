"""
Representation Extractor for Probe V3.
Extracts Layer 12 AST sink-region pooled representations from full pretrained Mistral-7B.
Enforces strict preservation: no mock models, miniature fixtures, or synthetic vectors.
"""

import os
import sys
from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch

# Ensure extraction module is discoverable
EXTRACT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if EXTRACT_PATH not in sys.path:
    sys.path.insert(0, EXTRACT_PATH)

from extraction.ast_sink_extractor import (
    extract_ast_sink_regions,
    pool_ast_sink_representation
)

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
TARGET_LAYER = 12
HIDDEN_DIM = 4096

class RepresentationExtractor:
    """Manages Layer 12 representation extraction using Mistral-7B."""

    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = MODEL_NAME
        self.target_layer = TARGET_LAYER
        self.hidden_dim = HIDDEN_DIM

    def load_pretrained_model(self, model_name: str = MODEL_NAME, device_map: str = "auto"):
        """Loads the full pretrained Mistral-7B-Instruct-v0.3 in 4-bit NF4 precision."""
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU is required to load pretrained Mistral-7B in 4-bit NF4 precision. "
                "Synthetic fallbacks and miniature fixtures are strictly prohibited."
            )

        from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        config = AutoConfig.from_pretrained(model_name)
        assert config.hidden_size == HIDDEN_DIM, f"Expected hidden size {HIDDEN_DIM}, got {config.hidden_size}"
        assert config.num_hidden_layers == 32, f"Expected 32 layers, got {config.num_hidden_layers}"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=torch.bfloat16
        )
        self.model.eval()

    def extract_from_code(self, code: str) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Extracts 4096-dim Layer 12 AST sink pooled representation from raw Python code.
        
        Returns:
            (embedding_vector, metadata_dict)
            If model is not loaded, returns (None, error_metadata) without mock fallback.
        """
        if self.model is None or self.tokenizer is None:
            return None, {
                "status": "UNAVAILABLE",
                "reason": (
                    "Mistral-7B model is not initialized. Loading full 7B weights requires GPU. "
                    "In CPU-only environments, supply pre-extracted representations or use AST-only analysis."
                )
            }

        # 1. AST Sink extraction
        sink_info = extract_ast_sink_regions(code)

        # 2. Tokenize with offset mapping
        inputs = self.tokenizer(
            code,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=2048
        ).to(self.model.device)

        offsets = inputs.pop("offset_mapping")[0].tolist()
        mask = inputs["attention_mask"][0]

        # 3. Model forward pass
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # outputs.hidden_states[12] corresponds to Layer 12 output
        h_layer12 = outputs.hidden_states[self.target_layer][0] # (seq_len, 4096)

        # 4. AST Sink Pooling
        sink_vec, tokens_used, fallback_used = pool_ast_sink_representation(
            hidden_states=h_layer12,
            offset_mapping=offsets,
            attention_mask=mask,
            sink_info=sink_info
        )

        arr = sink_vec.cpu().to(torch.float32).numpy()
        assert arr.shape == (HIDDEN_DIM,), f"Wrong vector shape: {arr.shape}"

        return arr, {
            "status": "EXTRACTED",
            "layer": self.target_layer,
            "tokens_pooled": len(tokens_used),
            "fallback_used": fallback_used
        }

