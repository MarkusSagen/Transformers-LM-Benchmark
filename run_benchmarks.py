import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import transformers
from transformers import (
    AutoConfig,
    PyTorchBenchmark,
    PyTorchBenchmarkArguments,
)
from transformers.configuration_utils import PretrainedConfig

transformers.set_seed(42)
BASE_DIR = "data"


def save_to_csv(result_dict, filename):
    """Save benchmark to CSV."""

    Path(BASE_DIR).mkdir(parents=True, exist_ok=True)
    with open(filename, mode="w") as csv_file:

        fieldnames = ["model", "batch_size", "sequence_length"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames + ["result"])
        writer.writeheader()

        model_names = list(result_dict.keys())
        for model_name in model_names:
            result_dict_model = result_dict[model_name]["result"]
            for bs in result_dict_model:
                for ss in result_dict_model[bs]:
                    result_model = result_dict_model[bs][ss]
                    writer.writerow(
                        {
                            "model": model_name,
                            "batch_size": bs,
                            "sequence_length": ss,
                            "result": (
                                "{}"
                                if not isinstance(result_model, float)
                                else "{:.4f}"
                            ).format(result_model),
                        }
                    )


@dataclass
class BenchmarkWrapper:
    """Dataclass params for Huggingface's PyTorchBenchmark"""

    configs: List[str]
    models: List[str]
    sequence_lengths: List[str]
    batch_sizes: List[str]


def run_benchmark(
    benchmark_config: BenchmarkWrapper, result_name: str, fp16=False, training=False
):
    """Run Huggingface's PyTorch benchmarks.

    args:
    benchmark_config: BenchmarkWrapper
    result_name: str = Name prepended (without spaces or special characters) for the saved result
    fp16: bool = If halved precision should be used or not
    """

    configs = benchmark_config.configs
    models = benchmark_config.models
    sequence_lengths = benchmark_config.sequence_lengths
    batch_sizes = benchmark_config.batch_sizes

    benchmark_args = PyTorchBenchmarkArguments(
        models=models,
        sequence_lengths=sequence_lengths,
        batch_sizes=batch_sizes,
        fp16=fp16,
        training=training,
    )
    benchmark = PyTorchBenchmark(configs=configs, args=benchmark_args)
    results = benchmark.run()

    name = str(result_name)
    name += "_fp" if fp16 else "_no_fp"
    save_to_csv(
        results.time_inference_result, f"{BASE_DIR}/{name}_time_inference_result.csv"
    )
    save_to_csv(
        results.memory_inference_result,
        f"{BASE_DIR}/{name}_memory_inference_result.csv",
    )
    save_to_csv(results.time_train_result, f"{BASE_DIR}/{name}_time_train_result.csv")
    save_to_csv(
        results.memory_train_result, f"{BASE_DIR}/{name}_memory_train_result.csv"
    )

    return results


"""
# Test

config_4_layers_bert = AutoConfig.from_pretrained("bert-base-uncased", num_hidden_layers=4)
res = BenchmarkWrapper(
    configs=[config_4_layers_bert],
    models=["Bert-4-Layers"],
    sequence_lengths=[512],
    batch_sizes=[8],
)
result = run_benchmark(res, result_name="test_bert", fp16=False, training=True)
exit(1)
"""


# Standard
bert_base = AutoConfig.from_pretrained("bert-base-cased")
bert_large = AutoConfig.from_pretrained("bert-large-cased")
roberta_base = AutoConfig.from_pretrained("roberta-base")
roberta_large = AutoConfig.from_pretrained("roberta-large")

# distilled
distil_bert = AutoConfig.from_pretrained("distilbert-base-uncased")
# conv_bert = AutoConfig.from_pretrained("YituTech/conv-bert-base")
minilm = AutoConfig.from_pretrained("microsoft/MiniLM-L12-H384-uncased")
minilm_multilingual = AutoConfig.from_pretrained(
    "microsoft/Multilingual-MiniLM-L12-H384"
)

# Multilingual
mbert_base = AutoConfig.from_pretrained("bert-base-multilingual-cased")
xlmr_base = AutoConfig.from_pretrained("xlm-roberta-base")
xlmr_large = AutoConfig.from_pretrained("xlm-roberta-large")

# generative
gpt2 = AutoConfig.from_pretrained("gpt2")
gpt_neo_1B = AutoConfig.from_pretrained("EleutherAI/gpt-neo-1.3B")
gpt_neo_2B = AutoConfig.from_pretrained("EleutherAI/gpt-neo-2.7B")
bart_large = AutoConfig.from_pretrained("facebook/bart-large")
mbart_large = AutoConfig.from_pretrained("facebook/mbart-large-cc25")

# NOTE: Generative no training - does not work with TF test
# t5_base = AutoConfig.from_pretrained("t5-base")
# t5_large = AutoConfig.from_pretrained("t5-large")
# mt5_base = AutoConfig.from_pretrained("google/mt5-small")
# mt5_large = AutoConfig.from_pretrained("google/mt5-large")

# Efficient
big_bird_base = AutoConfig.from_pretrained("google/bigbird-roberta-base")
longformer_base_4096 = AutoConfig.from_pretrained("allenai/longformer-base-4096")
longformer_large_4096 = AutoConfig.from_pretrained("allenai/longformer-large-4096")
transformer_xl = AutoConfig.from_pretrained("transfo-xl-wt103")
# 18 layers default

# Efficient no training
# NOTE: Funnel transformers causes errors
# funnel_base = AutoConfig.from_pretrained("funnel-transformer/small-base")
# funnel_large = AutoConfig.from_pretrained("funnel-transformer/large-base")
reformer_enwik8 = AutoConfig.from_pretrained("google/reformer-enwik8")


# Shared for several models
sequence_lengths: List[str] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
batch_sizes: List[str] = [1, 2, 4, 8, 16, 32]
efficient_sequence_lengths: List[str] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

# Regular transformers that can be trained
configs: List[str]
configs = [
    bert_base,
    bert_large,
    roberta_base,
    roberta_large,
    distil_bert,
    minilm,
    minilm_multilingual,
    mbert_base,
    xlmr_base,
    xlmr_large,
    gpt2,
]
models = [
    "BERT_base",
    "BERT_large",
    "RoBERTa_base",
    "RoBERTa_large",
    "dist_bert",
    "MiniLM",
    "MiniLM_multilingual",
    "mBERT_cased",
    "XLM-R_base",
    "XLM-R_large",
    "GPT-2",
]
benchmark_config = BenchmarkWrapper(
    configs=configs,
    models=models,
    sequence_lengths=sequence_lengths,
    batch_sizes=batch_sizes,
)
run_benchmark(benchmark_config, result_name="regular", fp16=False, training=True)

# Inference
configs: List[str]
configs = [
    bert_base,
    bert_large,
    roberta_base,
    roberta_large,
    distil_bert,
    minilm,
    minilm_multilingual,
    mbert_base,
    xlmr_base,
    xlmr_large,
    gpt2,
    bart_large,
    mbart_large,
]
models = [
    "BERT_base",
    "BERT_large",
    "RoBERTa_base",
    "RoBERTa_large",
    "dist_bert",
    "MiniLM",
    "MiniLM_multilingual",
    "mBERT_cased",
    "XLM-R_base",
    "XLM-R_large",
    "GPT-2",
    "BART_large",
    "mBART_large",
]
benchmark_config = BenchmarkWrapper(
    configs=configs,
    models=models,
    sequence_lengths=sequence_lengths,
    batch_sizes=batch_sizes,
)
run_benchmark(benchmark_config, result_name="regular", fp16=False, training=False)


# Efficient transformer models
# Extends/augments regular models
configs = [
    big_bird_base,
    longformer_base_4096,
    longformer_large_4096,
    transformer_xl,
]
models = [
    "BigBird",  # roberta-base
    "Longformer_base",  # 4096 seq len
    "Longformer_large",
    "Transformer-XL",
]
benchmark_config = BenchmarkWrapper(
    configs=configs,
    models=models,
    sequence_lengths=efficient_sequence_lengths,
    batch_sizes=batch_sizes,
)
run_benchmark(
    benchmark_config, result_name="efficient_training", fp16=False, training=True
)

# Inference
configs = [
    big_bird_base,
    longformer_base_4096,
    longformer_large_4096,
    transformer_xl,
]
models = [
    "BigBird",  # roberta-base
    "Longformer_base",  # 4096 seq len
    "Longformer_large",
    "Transformer-XL",
]
benchmark_config = BenchmarkWrapper(
    configs=configs,
    models=models,
    sequence_lengths=efficient_sequence_lengths,
    batch_sizes=batch_sizes,
)
run_benchmark(
    benchmark_config, result_name="efficient_training", fp16=False, training=False
)


# Efficient models
# THESE MODELS CAN'T HANDLE Train = True
configs = [
    # funnel_base,
    # funnel_large,
    reformer_enwik8,
]
models = [
    # "Funnel-transformer-small",
    # "Funnel-transformer-large",
    "Reformer",
]
benchmark_config = BenchmarkWrapper(
    configs=configs,
    models=models,
    sequence_lengths=efficient_sequence_lengths,
    batch_sizes=batch_sizes,
)
run_benchmark(benchmark_config, result_name="efficient_new", fp16=False, training=False)
