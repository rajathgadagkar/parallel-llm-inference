# Parallel LLM Inference

A two-worker pipeline-parallel inference engine for small pretrained decoder-only language models, built to explore inter-process communication, per-request KV-cache isolation, flow control, and fault recovery.

## Overview

Instead of running a full language model inside a single process, this project partitions a pretrained decoder-only Transformer across two independent worker processes communicating over gRPC/TCP:

- **Coordinator**: Tokenizes prompts, manages concurrent requests, enforces bounded queues and timeouts, selects generated tokens, tracks request/step/epoch state, and records performance metrics.
- **Worker 1**: Runs token and position embeddings along with the first partition of Transformer layers, maintains the KV cache for its layers, and streams hidden-state activations to Worker 2.
- **Worker 2**: Runs the remaining Transformer layers and language-modeling head, maintains the KV cache for its layers, and returns logits/next-token outputs to the coordinator.

All pipeline runs are evaluated against a single-process baseline using the same model weights, tokenizer, prompts, and deterministic greedy decoding settings.

## Goals & Key Capabilities

- **Model Partitioning over RPC**: Split a pretrained causal LM (e.g., DistilGPT-2 or another compatible small checkpoint) across two worker processes using typed Protocol Buffers and gRPC over TCP.
- **Per-Request Ordering & KV-Cache Isolation**: Serve concurrent requests while preserving token ordering and isolating KV-cache state per request.
- **Flow Control & Backpressure**: Enforce bounded request and inter-stage queues, explicit overload rejection, finite RPC deadlines, and optional micro-batching once single-request decoding is verified.
- **Fault Tolerance & Recovery**: Handle worker termination and restart with epoch tracking, stale-response rejection, bounded retries, and KV-cache reconstruction via prompt and committed-token history replay.
- **Benchmarking & Verification**: Verify output agreement against the single-process baseline and measure time to first token (TTFT), median/p95 token latency, throughput (tokens/s and requests/s), queue wait time, activation vs. control-message bytes, and recovery time.

## Planned Architecture & Layout

```text
parallel-llm-inference/
├── proto/                 # Protocol Buffer service and tensor/activation schemas
├── src/
│   ├── baseline/          # Single-process reference inference implementation
│   ├── coordinator/       # Request scheduler, tokenizer, retry/epoch manager, metrics
│   └── worker/            # Partitioned model stages, KV-cache store, gRPC services
├── scripts/               # Launch scripts, benchmarks, and failure-injection helpers
├── tests/                 # Correctness, concurrency, and recovery tests (pytest)
├── requirements.txt
└── README.md
```

## Tech Stack

- **Language**: Python 3.11+
- **ML / Model Execution**: PyTorch, Hugging Face `transformers`
- **RPC & Serialization**: gRPC (`grpcio`, `grpcio-tools`), Protocol Buffers
- **Analysis & Testing**: NumPy, pandas, Matplotlib, `pytest`

## Getting Started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Contributors

- Rajath Gadagkar
- Sai Kiran
