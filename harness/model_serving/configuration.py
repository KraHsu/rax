"""Declarative, credential-free inference configuration."""
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServeConfig:
    model: str
    revision: str
    image: str
    served_model: str = 'deepseek-flash'
    tensor_parallel_size: int = 4
    max_model_len: int = 1048576
    max_num_seqs: int = 4
    max_num_batched_tokens: int = 4096
    gpu_memory_utilization: float = 0.90
    port: int = 8000
    host: str = '0.0.0.0'
    reasoning_effort: int = 100
    enforce_eager: bool = True
    compilation_config: dict | None = None
    speculative_config: dict | None = None
    engram_use_thp: bool = False
    cpu_threads: int = 1
    extra_args: tuple[str, ...] = ()

    def __post_init__(self):
        if not re.fullmatch(r'[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+', self.model):
            raise ValueError('model must be an owner/repository identifier')
        if not re.fullmatch(r'[0-9a-f]{40}', self.revision):
            raise ValueError('revision must pin a full Hugging Face commit SHA')
        if not re.fullmatch(r'.+@sha256:[0-9a-f]{64}', self.image):
            raise ValueError('image must pin its OCI SHA256 digest')
        for name in ('tensor_parallel_size', 'max_model_len', 'max_num_seqs', 'max_num_batched_tokens', 'cpu_threads'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not 1 <= self.port <= 65535 or not 0 < self.gpu_memory_utilization < 1:
            raise ValueError('Invalid port or GPU memory fraction')
        if not 1 <= self.reasoning_effort <= 100:
            raise ValueError('reasoning_effort must be within [1,100]')
        if self.enforce_eager and self.compilation_config:
            raise ValueError('CUDA graph compilation requires enforce_eager=false')
        if self.speculative_config and self.speculative_config.get('rejection_sample_method') == 'synthetic':
            raise ValueError('Synthetic acceptance is not valid for real inference')

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if 'extra_args' in data:
            data['extra_args'] = tuple(data['extra_args'])
        return cls(**data)

    def arguments(self, weights):
        # The api_server module does not promote positional model_tag to model
        # as the `vllm serve` CLI does. Select the checkpoint explicitly.
        arguments = ['--model', str(weights), '--served-model-name', self.served_model,
                '--host', self.host, '--port', str(self.port),
                '--tensor-parallel-size', str(self.tensor_parallel_size),
                '--tokenizer-mode', 'deepseek_v41', '--reasoning-parser', 'deepseek_v41',
                '--tool-call-parser', 'deepseek_v41', '--enable-auto-tool-choice',
                '--engram-config', json.dumps({'cpu_offload': True, **({'use_thp': True} if self.engram_use_thp else {})}),
                '--max-model-len', str(self.max_model_len), '--max-num-seqs', str(self.max_num_seqs),
                '--max-num-batched-tokens', str(self.max_num_batched_tokens),
                '--gpu-memory-utilization', str(self.gpu_memory_utilization),
                '--default-chat-template-kwargs', json.dumps({'thinking': True, 'reasoning_effort': self.reasoning_effort}),
                '--enable-prefix-caching']
        if self.enforce_eager:
            arguments.append('--enforce-eager')
        if self.compilation_config:
            arguments += ['--compilation-config', json.dumps(self.compilation_config)]
        if self.speculative_config:
            arguments += ['--speculative-config', json.dumps(self.speculative_config)]
        return [*arguments, *self.extra_args]
