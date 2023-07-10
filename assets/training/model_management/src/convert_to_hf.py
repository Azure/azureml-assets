# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run Model preprocessor module."""

import argparse
from pathlib import Path
import os
import gc
import json
import math
import shutil
import warnings
from tempfile import TemporaryDirectory
import subprocess
import torch
import subprocess

from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

try:
    from transformers import LlamaTokenizerFast
except ImportError as e:
    warnings.warn(e)
    warnings.warn(
        "The converted tokenizer will be the `slow` tokenizer. To use the fast, update your `tokenizers` library and re-run the tokenizer conversion"
    )
    LlamaTokenizerFast = None


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True, help="Model input path")
    parser.add_argument(
        "--hf-model-output-dir",
        type=Path,
        required=True,
        help="Output HF model",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["7B", "13B", "30B", "65B", "tokenizer_only"],
        help="Model size",
        default="7B"
    )
    parser.add_argument(
        "--convert-to-hf",
        type=bool,
        required=True,
        help="Parameter to identify whether we need to convert in HF format or not.",
        default="True"
    )
    return parser


def find_and_unzip_zip_files(folder_path, destination_path):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith('.zip'):
            try:
                subprocess.run(["unzip", file_path, "-d", destination_path])
                print("Extracted succesfully")
            except Exception as e:
                raise Exception(f"Error extracting file: {e}")

def search_folder(directory, target_folder):
    # Traverse the directory tree recursively
    for root, dirs, files in os.walk(directory):
        # Check if the target folder is found
        if target_folder in dirs:
            # Return the absolute path of the target folder
            return os.path.join(root, target_folder)
    
    # Target folder not found
    return None

INTERMEDIATE_SIZE_MAP = {
    "7B": 11008,
    "13B": 13824,
    "30B": 17920,
    "65B": 22016,
}
NUM_SHARDS = {
    "7B": 1,
    "13B": 2,
    "30B": 4,
    "65B": 8,
}


def compute_intermediate_size(n):
    return int(math.ceil(n * 8 / 3) + 255) // 256 * 256


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)


def write_model(model_path, input_base_path, model_size):
    os.makedirs(model_path, exist_ok=True)
    tmp_model_path = os.path.join(model_path, "tmp")
    os.makedirs(tmp_model_path, exist_ok=True)

    params = read_json(os.path.join(input_base_path, "params.json"))
    num_shards = NUM_SHARDS[model_size]
    n_layers = params["n_layers"]
    n_heads = params["n_heads"]
    n_heads_per_shard = n_heads // num_shards
    dim = params["dim"]
    dims_per_head = dim // n_heads
    base = 10000.0
    inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))

    # permute for sliced rotary
    def permute(w):
        return w.view(n_heads, dim // n_heads // 2, 2, dim).transpose(1, 2).reshape(dim, dim)

    print(f"Fetching all parameters from the checkpoint at {input_base_path}.")
    # Load weights
    if model_size == "7B":
        # Not sharded
        # (The sharded implementation would also work, but this is simpler.)
        loaded = torch.load(os.path.join(input_base_path, "consolidated.00.pth"), map_location="cpu")
    else:
        # Sharded
        loaded = [
            torch.load(os.path.join(input_base_path, f"consolidated.{i:02d}.pth"), map_location="cpu")
            for i in range(num_shards)
        ]
    param_count = 0
    index_dict = {"weight_map": {}}
    for layer_i in range(n_layers):
        filename = f"pytorch_model-{layer_i + 1}-of-{n_layers + 1}.bin"
        if model_size == "7B":
            # Unsharded
            state_dict = {
                f"model.layers.{layer_i}.self_attn.q_proj.weight": permute(
                    loaded[f"layers.{layer_i}.attention.wq.weight"]
                ),
                f"model.layers.{layer_i}.self_attn.k_proj.weight": permute(
                    loaded[f"layers.{layer_i}.attention.wk.weight"]
                ),
                f"model.layers.{layer_i}.self_attn.v_proj.weight": loaded[f"layers.{layer_i}.attention.wv.weight"],
                f"model.layers.{layer_i}.self_attn.o_proj.weight": loaded[f"layers.{layer_i}.attention.wo.weight"],
                f"model.layers.{layer_i}.mlp.gate_proj.weight": loaded[f"layers.{layer_i}.feed_forward.w1.weight"],
                f"model.layers.{layer_i}.mlp.down_proj.weight": loaded[f"layers.{layer_i}.feed_forward.w2.weight"],
                f"model.layers.{layer_i}.mlp.up_proj.weight": loaded[f"layers.{layer_i}.feed_forward.w3.weight"],
                f"model.layers.{layer_i}.input_layernorm.weight": loaded[f"layers.{layer_i}.attention_norm.weight"],
                f"model.layers.{layer_i}.post_attention_layernorm.weight": loaded[f"layers.{layer_i}.ffn_norm.weight"],
            }
        else:
            # Sharded
            # Note that attention.w{q,k,v,o}, feed_fordward.w[1,2,3], attention_norm.weight and ffn_norm.weight share
            # the same storage object, saving attention_norm and ffn_norm will save other weights too, which is
            # redundant as other weights will be stitched from multiple shards. To avoid that, they are cloned.

            state_dict = {
                f"model.layers.{layer_i}.input_layernorm.weight": loaded[0][
                    f"layers.{layer_i}.attention_norm.weight"
                ].clone(),
                f"model.layers.{layer_i}.post_attention_layernorm.weight": loaded[0][
                    f"layers.{layer_i}.ffn_norm.weight"
                ].clone(),
            }
            state_dict[f"model.layers.{layer_i}.self_attn.q_proj.weight"] = permute(
                torch.cat(
                    [
                        loaded[i][f"layers.{layer_i}.attention.wq.weight"].view(n_heads_per_shard, dims_per_head, dim)
                        for i in range(num_shards)
                    ],
                    dim=0,
                ).reshape(dim, dim)
            )
            state_dict[f"model.layers.{layer_i}.self_attn.k_proj.weight"] = permute(
                torch.cat(
                    [
                        loaded[i][f"layers.{layer_i}.attention.wk.weight"].view(n_heads_per_shard, dims_per_head, dim)
                        for i in range(num_shards)
                    ],
                    dim=0,
                ).reshape(dim, dim)
            )
            state_dict[f"model.layers.{layer_i}.self_attn.v_proj.weight"] = torch.cat(
                [
                    loaded[i][f"layers.{layer_i}.attention.wv.weight"].view(n_heads_per_shard, dims_per_head, dim)
                    for i in range(num_shards)
                ],
                dim=0,
            ).reshape(dim, dim)

            state_dict[f"model.layers.{layer_i}.self_attn.o_proj.weight"] = torch.cat(
                [loaded[i][f"layers.{layer_i}.attention.wo.weight"] for i in range(num_shards)], dim=1
            )
            state_dict[f"model.layers.{layer_i}.mlp.gate_proj.weight"] = torch.cat(
                [loaded[i][f"layers.{layer_i}.feed_forward.w1.weight"] for i in range(num_shards)], dim=0
            )
            state_dict[f"model.layers.{layer_i}.mlp.down_proj.weight"] = torch.cat(
                [loaded[i][f"layers.{layer_i}.feed_forward.w2.weight"] for i in range(num_shards)], dim=1
            )
            state_dict[f"model.layers.{layer_i}.mlp.up_proj.weight"] = torch.cat(
                [loaded[i][f"layers.{layer_i}.feed_forward.w3.weight"] for i in range(num_shards)], dim=0
            )

        state_dict[f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq"] = inv_freq
        for k, v in state_dict.items():
            index_dict["weight_map"][k] = filename
            param_count += v.numel()
        torch.save(state_dict, os.path.join(tmp_model_path, filename))

    filename = f"pytorch_model-{n_layers + 1}-of-{n_layers + 1}.bin"
    if model_size == "7B":
        # Unsharded
        state_dict = {
            "model.embed_tokens.weight": loaded["tok_embeddings.weight"],
            "model.norm.weight": loaded["norm.weight"],
            "lm_head.weight": loaded["output.weight"],
        }
    else:
        state_dict = {
            "model.norm.weight": loaded[0]["norm.weight"],
            "model.embed_tokens.weight": torch.cat(
                [loaded[i]["tok_embeddings.weight"] for i in range(num_shards)], dim=1
            ),
            "lm_head.weight": torch.cat([loaded[i]["output.weight"] for i in range(num_shards)], dim=0),
        }

    for k, v in state_dict.items():
        index_dict["weight_map"][k] = filename
        param_count += v.numel()
    torch.save(state_dict, os.path.join(tmp_model_path, filename))

    # Write configs
    index_dict["metadata"] = {"total_size": param_count * 2}
    write_json(index_dict, os.path.join(tmp_model_path, "pytorch_model.bin.index.json"))

    config = LlamaConfig(
        hidden_size=dim,
        intermediate_size=compute_intermediate_size(dim),
        num_attention_heads=params["n_heads"],
        num_hidden_layers=params["n_layers"],
        rms_norm_eps=params["norm_eps"],
    )
    config.save_pretrained(tmp_model_path)

    # Make space so we can load the model properly now.
    del state_dict
    del loaded
    gc.collect()

    print("Loading the checkpoint in a Llama model.")
    model = LlamaForCausalLM.from_pretrained(tmp_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    # Avoid saving this as part of the config.
    del model.config._name_or_path

    print("Saving in the Transformers format.")
    model.save_pretrained(model_path)
    shutil.rmtree(tmp_model_path)


def write_tokenizer(tokenizer_path, input_tokenizer_path):
    # Initialize the tokenizer based on the `spm` model
    tokenizer_class = LlamaTokenizer if LlamaTokenizerFast is None else LlamaTokenizerFast
    print(f"Saving a {tokenizer_class.__name__} to {tokenizer_path}.")
    tokenizer = tokenizer_class(input_tokenizer_path)
    tokenizer.save_pretrained(tokenizer_path)


def run():
    """Run preprocess."""
    parser = _get_parser()
    args, _ = parser.parse_known_args()

    print("Args received", args)
    model_path = args.model_path
    hf_model_output_dir = args.hf_model_output_dir
    model_size = args.model_size
    convert_to_hf = args.convert_to_hf

    if convert_to_hf:
        with TemporaryDirectory(dir=hf_model_output_dir) as temp_dir:
            find_and_unzip_zip_files(model_path, temp_dir)
            if model_size != "tokenizer_only":
                print("Converting to HF")
                target_folder = "models"
                models_weight_path = search_folder(temp_dir, target_folder)
                write_model(
                    model_path=hf_model_output_dir,
                    input_base_path=os.path.join(models_weight_path, model_size),
                    model_size=model_size,
                )
            spm_path = os.path.join(models_weight_path, "tokenizer.model")
            write_tokenizer(hf_model_output_dir, spm_path)
    else:
        #Simply copy the model to output directory
        shutil.copytree(model_path, hf_model_output_dir)

if __name__ == "__main__":
    run()
