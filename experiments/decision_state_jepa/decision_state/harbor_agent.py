"""Harbor custom agent evaluating actual SALM LoRA/Decision-State checkpoints.

Uses one identical terminal-agent implementation for both experimental arms.
Harbor's sandbox verifier, not this module, decides whether a task passed.
Requires 'harbor' to be installed only for benchmark runs.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from .runtime import load_checkpoint, generate_decision
from .terminal_protocol import (initial_events, compile_prompt, parse_tool_call,
                                command_for_tool, forbidden_command)


class SALMTerminalAgent(BaseAgent):
    """GPU inference runs on host; all generated commands run in Harbor sandbox."""

    @staticmethod
    def name() -> str:
        return 'salm-terminal-agent'

    def version(self) -> str:
        return '0.1.0'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        checkpoint = os.environ.get('SALM_DECISION_CHECKPOINT', '').strip()
        if not checkpoint:
            raise RuntimeError('Set SALM_DECISION_CHECKPOINT to a trained checkpoint path')
        self.checkpoint = str(Path(checkpoint).expanduser().resolve())
        self.max_turns = min(max(int(os.environ.get('SALM_TERMINAL_MAX_TURNS','24')),1),150)
        self.max_new_tokens = min(max(int(os.environ.get('SALM_TERMINAL_NEW_TOKENS','512')),32),2048)
        self.command_timeout = min(max(int(os.environ.get('SALM_TERMINAL_COMMAND_TIMEOUT','90')),5),600)
        self.max_output_chars = min(max(int(os.environ.get('SALM_TERMINAL_OUTPUT_CHARS','12000')),2000),60000)
        self._bundle = None

    async def setup(self, environment: BaseEnvironment) -> None:
        # No model weight or benchmark secret is copied into task containers.
        # Harbor provisions the isolated task and applies its test verifier.
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    async def run(self, instruction: str, environment: BaseEnvironment,
                  context: AgentContext) -> None:
        if self._bundle is None:
            self._bundle = await asyncio.to_thread(load_checkpoint, self.checkpoint)
        model, tokenizer, conditioner, cfg = self._bundle
        events = initial_events(instruction)
        log_file = self.logs_dir / 'salm_terminal_trace.jsonl'
        bad_responses = 0
        for turn in range(self.max_turns):
            prompt = compile_prompt(events)
            text = await asyncio.to_thread(
                generate_decision, model, tokenizer, conditioner, prompt,
                max_new_tokens=self.max_new_tokens,
                max_context=min(int(cfg['max_seq_length']), 16384),
            )
            call = parse_tool_call(text)
            if call is None:
                kind = 'answer' if 'kind=answer' in text or 'Final' in text else 'reasoning'
            else:
                kind = 'tool_call'
            events.append({'role':'assistant','kind':kind,'text':text[:self.max_output_chars]})
            record: dict[str,Any] = {'turn':turn,'model_mode':cfg['mode'],
                                     'assistant':text[:self.max_output_chars],
                                     'tool_call':call}
            if call is None:
                bad_responses += 1
                if kind=='answer' or bad_responses>=2:
                    record['status'] = 'final_or_no_tool'
                    with log_file.open('a',encoding='utf-8') as fp:
                        fp.write(json.dumps(record,ensure_ascii=False)+'\n')
                    break
                events.append({'role':'user','kind':'user',
                               'text':'If the task is not complete, use a tool call. Otherwise give a final answer.'})
            else:
                bad_responses = 0
                command = command_for_tool(call)
                if command is None:
                    observation = 'Tool is not supported. Use Bash/Read/Write/Edit/Grep/Glob.'
                    record['status'] = 'unsupported_tool'
                elif forbidden_command(command):
                    observation = 'Denied: hidden verifier or solution files are off limits.'
                    record['status'] = 'forbidden_command'
                else:
                    try:
                        output = await environment.exec(command=command,
                              timeout_sec=self.command_timeout)
                        observation = f'Exit code: {output.return_code}\n' \
                                      f'STDOUT:\n{output.stdout or ""}\nSTDERR:\n{output.stderr or ""}'
                        record['status'] = 'ran'
                        record['return_code'] = output.return_code
                    except Exception as exc:
                        observation = f'Tool execution failed: {type(exc).__name__}: {str(exc)[:300]}'
                        record['status'] = 'exec_error'
                # Never exceed the model context with enormous tool outputs.
                observation = observation[:self.max_output_chars]
                events.append({'role':'tool','kind':'tool','text':observation})
                record['observation'] = observation
            with log_file.open('a',encoding='utf-8') as fp:
                fp.write(json.dumps(record,ensure_ascii=False)+'\n')
        # Let Harbor's *official verifier* grade the sandbox after this returns.
