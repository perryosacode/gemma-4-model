#!/usr/bin/env python3
"""Interactive REPL to verify Gemma4 q4f16_0 in mlc-llm on a single GPU.

The `mlc_llm chat` CLI can't set gpu_memory_utilization, and q4f16_0 (13.5 GB weights)
needs ~0.96 to fit alongside the KV cache + temp buffer on a 16 GB card, so we drive
MLCEngine directly. Each turn is sent fresh (no growing history) to stay within the
2048-token context the q4 lib was compiled for.
"""
import sys
from mlc_llm import MLCEngine
from mlc_llm.serve.config import EngineConfig

MODEL = "/tmp/gemma4-q4f16_0-MLC-weights"
MODEL_LIB = "/tmp/gemma4-q4f16_0-lean.so"  # compiled with prefill 1024 / ctx 1024 for 16 GB fit

# q4f16_0 weights are 13.5 GB. On a 16 GB card (with ~0.9 GB used by the desktop and ~0.4 GB
# CUDA overhead) this is tight, so the lib is compiled with prefill 1024 / ctx 1024 to keep the
# temp buffer + KV cache small. gpu_memory_utilization is a fraction of TOTAL memory; 0.93 keeps
# the budget within real free memory while leaving room for weights + temp + KV.
engine = MLCEngine(
    model=MODEL,
    model_lib=MODEL_LIB,
    mode="local",
    engine_config=EngineConfig(gpu_memory_utilization=0.93),
)
print("\nGemma4 q4f16_0 ready. Type a message; 'exit' or Ctrl-D to quit.\n")

try:
    while True:
        try:
            user = input("You: ").strip()
        except EOFError:
            break
        if user.lower() in ("exit", "quit", "/exit"):
            break
        if not user:
            continue
        print("Gemma4: ", end="", flush=True)
        for resp in engine.chat.completions.create(
            messages=[{"role": "user", "content": user}],
            stream=True, max_tokens=512, temperature=0.7, top_p=0.95,
        ):
            for c in resp.choices:
                if c.delta.content:
                    sys.stdout.write(c.delta.content)
                    sys.stdout.flush()
        print("\n")
finally:
    engine.terminate()
    print("bye")
