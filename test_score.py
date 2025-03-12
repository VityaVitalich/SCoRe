import argparse
import yaml
import torch
import torch.nn as nn

from utils.generation_utils import load_config
from accelerate.utils import extract_model_from_parallel

from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    #✅ 1. Load config
    config = load_config(args.config_path)
    # config = {
    #     'model_path': 'Qwen/Qwen2.5-Math-1.5B-Instruct',
    #     'cache_dir': './'
    # }

    # ✅ 2. Initialize Accelerator (with FSDP/DeepSpeed)
    accelerator = Accelerator()
   # print(f"\n🚀 [INFO] Using Accelerator with {accelerator.state.fsdp_plugin}\n")

    # ✅ 3. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_path"], use_fast=True, cache_dir=config["cache_dir"]
    )

    # ✅ 4. Load Model
    print(f"🔄 Loading model from: {config['model_path']}...")
    model = AutoModelForCausalLM.from_pretrained(
        config["model_path"],
        torch_dtype=torch.bfloat16,  # Match mixed precision settings
        cache_dir=config["cache_dir"]
    )


    # ✅ 5. Prepare model for distributed training
    model = accelerator.prepare_model(model)
    print("✅ Model wrapped with Accelerator")
    
    print("\n🔍 Model Parameter Shapes:")
    print("   🔹 embed_tokens:",  model.model.embed_tokens, model.model.embed_tokens.weight.shape)
    print("   🔹 lm_head:", model.lm_head, model.lm_head.weight.shape)

    # ✅ 6. Unwrap model to check if FSDP/DeepSpeed is correctly handling parameters
    unwrapped_model = accelerator.unwrap_model(model)
    print(unwrapped_model.__class__.__name__)

    # ✅ 8. Verify Embeddings & LM Head
    print("\n🔍 Model Parameter Shapes:")
    print("   🔹 embed_tokens:", unwrapped_model.model.embed_tokens.weight.shape)
    print("   🔹 lm_head:", unwrapped_model.lm_head.weight.shape)


    # ✅ 9. Ensure `lm_head` is correctly restored (not flattened)
    if len(unwrapped_model.lm_head.weight.shape) != 2:
        print("❌ ERROR: `lm_head` is incorrectly shaped (should be 2D)")
    else:
        print("✅ `lm_head` has correct shape")

    # ✅ 10. Ensure `embed_tokens` is not sharded (should be `[vocab_size, hidden_dim]`)
    if len(unwrapped_model.model.embed_tokens.weight.shape) != 2:
        print("❌ ERROR: `embed_tokens` is incorrectly shaped (should be 2D)")
    else:
        print("✅ `embed_tokens` has correct shape")

    # ✅ 11. Run a Test Generation
    prompt = "hello"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(accelerator.device)

    print("\n🚀 Running Test Generation...")
    with torch.no_grad():
       # with torch.distributed.fsdp.FullyShardedDataParallel.summon_full_params(model):
        print("\n🔍 Model Parameter Shapes:")
        print("   🔹 embed_tokens:", model.model.embed_tokens.weight.shape)
        print("   🔹 lm_head:", model.lm_head.weight.shape)
        output_ids = model.generate(inputs=input_ids, max_new_tokens=3)
    
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"📝 Generated Output: {output_text}")

    print("\n🎉 Model unwrapping test completed successfully!")

if __name__ == "__main__":
    main()
