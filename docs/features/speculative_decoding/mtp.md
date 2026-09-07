# MTP (Multi-Token Prediction)

MTP is a speculative decoding method where the target model includes native
multi-token prediction capability. Unlike draft-model-based methods, you do not
need to provide a separate draft model.

MTP is useful when:

- Your model natively supports MTP.
- You want model-based speculative decoding with minimal extra configuration.

## Gemma 4 Assistant Models

Gemma 4 assistant checkpoints use vLLM's Gemma 4 MTP path. They are not generic
draft models, even though they are passed through the `model` field in
`--speculative-config`.

Use `"method": "mtp"` when serving Gemma 4 with an assistant checkpoint:

```bash
vllm serve google/gemma-4-E2B-it \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --speculative-config '{"method":"mtp","model":"gg-hf-am/gemma-4-E2B-it-assistant","num_speculative_tokens":1}'
```

The E2B, E4B, 12B, 26B-A4B, and 31B Gemma 4 IT assistant checkpoints are supported.
Tower-based variants use `model_type: gemma4_assistant` and the encoder-free
Gemma 4 Unified variant (12B) uses `model_type: gemma4_unified_assistant`.
vLLM maps both to `Gemma4MTPModel` internally and wires the assistant layers
to share KV cache with the target model.

If an older vLLM release logs `SpeculativeConfig(method='draft_model', ...)`
for a Gemma 4 assistant checkpoint, that release is treating the assistant as a
generic draft model and may fail during initialization for multimodal Gemma 4
targets. Upgrade to a version with Gemma 4 MTP support instead.

## Offline Example

```python
from vllm import LLM, SamplingParams

prompts = ["The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="XiaomiMiMo/MiMo-7B-Base",
    tensor_parallel_size=1,
    speculative_config={
        "method": "mtp",
        "num_speculative_tokens": 1,
    },
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

## Online Example

```bash
vllm serve XiaomiMiMo/MiMo-7B-Base \
    --tensor-parallel-size 1 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

## Hybrid Mamba and GDN Models

Models that mix linear-attention (Mamba / Gated DeltaNet) layers with full
attention, such as Qwen3-Next, Qwen3.5, Qwen3.8 and Qwen3.8-Flash-Next, keep a
fixed-size recurrent state per request in the KV cache pool. With speculative
decoding, every Mamba KV cache group reserves `1 + num_speculative_tokens`
state blocks per request instead of one, so that rejected draft tokens can be
rolled back, and those blocks stay allocated for the lifetime of the request.

This changes how many requests fit in the pool. For Qwen3.8-27B (48 GDN layers
in 3 KV cache groups, `mamba_cache_mode="align"`, the default when prefix
caching is enabled) a request shorter than one attention block needs:

| `num_speculative_tokens` | Blocks per request       |
|--------------------------|--------------------------|
| 0                        | 1 attention + 3 × 1 = 4  |
| 1                        | 1 attention + 3 × 2 = 7  |
| 2                        | 1 attention + 3 × 3 = 10 |

The scheduler admits about `(num_gpu_blocks - 1) / blocks_per_request`
concurrent requests, whatever `--max-num-seqs` says. The MTP layer's weights and
the extra draft state also make each block slightly larger, so `num_gpu_blocks`
itself drops when MTP is enabled. On a small pool this can cap concurrency at a
handful of requests and make MTP slower than plain decoding at higher batch
sizes, even with a high acceptance rate.

To check, compare the startup line `Maximum concurrency for N tokens per
request` with and without `--speculative-config`. The figure is computed at
`--max-model-len`, so treat it as a ratio between the two runs rather than an
absolute concurrency. If the pool is the limit:

- Lower `--max-model-len` or the weight footprint (for example, a quantized
  checkpoint) to free KV cache memory.
- Reduce `num_speculative_tokens`.
- Disable speculative decoding for high-concurrency serving.

## Notes

- MTP only works for model families that support MTP in vLLM.
- `num_speculative_tokens` controls speculative depth. A small value like `1`
  is a good default to start with.
- If your model does not support MTP, use another method such as EAGLE or draft
  model speculation.
