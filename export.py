import shutil
from pathlib import Path

from onnxruntime_tools import optimizer
from transformers.convert_graph_to_onnx import convert

from tagc.model import Classification, get_tokenizer


def export():
    shutil.rmtree("onnx", ignore_errors=1)
    model = Classification.from_pretrained("model")
    model.base_model.save_pretrained("./bertBase")
    convert(
        framework="pt",
        model="bertBase",  # CHANGED: refer to custom model
        tokenizer=get_tokenizer(),  # <-- CHANGED: add tokenizer
        output=Path("onnx/bert-base-cased.onnx"),
        opset=12,
    )

    # # Mixed precision conversion for bert-base-cased model converted from Pytorch
    optimized_model = optimizer.optimize_model(
        "onnx/bert-base-cased.onnx",  # CHANGED: original `bert-base-cased.onnx` didn't point to right directory
        model_type="bert",
        num_heads=12,
        hidden_size=768,
    )
    optimized_model.convert_model_float32_to_float16()
    optimized_model.save_model_to_file("onnx/bert-base-cased.onnx")
