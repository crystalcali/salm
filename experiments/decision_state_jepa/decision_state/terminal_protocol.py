"""Portable Terminal-Bench 2.1 tool-call protocol.

A trained language model only proposes commands; Harbor executes them in the
per-task sandbox. No reference solutions, hidden verifier files, or benchmark
problems are copied into training data.
"""
from __future__ import annotations

import base64
import json
import re
import shlex
from typing import Any

from .data import serialize_event


POLICY = '''You are an autonomous software engineering agent inside an isolated terminal.
Fix the user's task by inspecting files, writing code, running commands and checking results.
Use the following exact assistant event format (one action at a time):
<assistant kind=tool_call>
<tool_call>{"name":"Bash","args":{"command":"pwd && ls -la"}}</tool_call>
</assistant>
The environment then returns a <tool kind=tool> observation.
For a final response, use <assistant kind=answer> followed by the explanation and </assistant>.
Never claim execution succeeded before observing terminal output. Do not read
hidden benchmark solutions, grader scripts, or /logs/verifier files.
'''


def initial_events(instruction: str) -> list[dict[str, str]]:
    return [
        {'role':'system','kind':'system','text':POLICY},
        {'role':'user','kind':'user','text':instruction},
    ]


def compile_prompt(events: list[dict[str,str]]) -> str:
    return ''.join(serialize_event(e) for e in events)


def _find_json_after(text: str, start: int):
    """Decode one JSON value despite surrounding assistant text."""
    d = json.JSONDecoder()
    for i in range(start, min(start+80,len(text))):
        if text[i] in '[{':
            try:
                val, _ = d.raw_decode(text[i:])
            except ValueError:
                continue
            return val
    return None


def parse_tool_call(text: str) -> dict | None:
    if not isinstance(text,str):
        return None
    # Our canonical serialization and NVIDIA trace format.
    for m in re.finditer(r'<tool_call>|Tool calls:\s*|<assistant kind=tool_call>',text,re.IGNORECASE):
        value = _find_json_after(text,m.end())
        if value is None:
            continue
        if isinstance(value,list) and value:
            value=value[0]
        if isinstance(value,dict):
            if 'function' in value and isinstance(value['function'],dict):
                func=value['function']
                return {'name':str(func.get('name','')),
                        'args':_as_dict(func.get('arguments',{}))}
            name=value.get('name') or value.get('tool_name')
            if name:
                return {'name':str(name),
                        'args':_as_dict(value.get('args',value.get('arguments',value.get('input',{}))))}
    # If the trained model produces a plain JSON tool command, accept it.
    stripped=text.strip()
    if stripped.startswith('{'):
        obj=_find_json_after(stripped,0)
        if isinstance(obj,dict) and (obj.get('name') or obj.get('tool_name')):
            return {'name':str(obj.get('name') or obj.get('tool_name')),
                    'args':_as_dict(obj.get('args',obj.get('arguments',obj.get('input',{}))))}
    return None


def _as_dict(value: Any) -> dict:
    if isinstance(value,dict):
        return value
    if isinstance(value,str):
        try:
            obj=json.loads(value)
            if isinstance(obj,dict):
                return obj
        except ValueError:
            pass
        return {'command':value}
    return {}


def _arg(args: dict, *keys: str) -> str:
    for key in keys:
        value=args.get(key)
        if isinstance(value,(str,int,float)) and str(value):
            return str(value)
    return ''


def _file_edit_script(path: str, *, old: str | None = None, new: str) -> str:
    """Use encoded literals to handle quotes, multiline edits, and unicode."""
    p64=base64.b64encode(path.encode()).decode()
    n64=base64.b64encode(new.encode()).decode()
    o64=base64.b64encode(old.encode()).decode() if old is not None else None
    header="import base64, pathlib\n" \
           "dec=lambda x: base64.b64decode(x).decode('utf-8')\n" \
           f"p=pathlib.Path(dec('{p64}'))\n"
    if old is None:
        script=header+f"p.parent.mkdir(parents=True, exist_ok=True)\np.write_text(dec('{n64}'))\n"
    else:
        script=header+f"s=p.read_text()\no=dec('{o64}')\nassert o in s, 'Old text was not found'\np.write_text(s.replace(o,dec('{n64}'),1))\n"
    return "python3 - <<'PY'\n"+script+"PY"


def command_for_tool(call: dict) -> str | None:
    """Translate common NVIDIA tool calls into a single sandbox command.

    Returns None for unsupported tools (agent receives an error observation).
    """
    name=str(call.get('name','')).lower().replace('-','').replace('_','')
    args=call.get('args') or {}
    if not isinstance(args,dict):
        return None
    if name in {'bash','shell','execute','exec','terminal','command','runcommand','functionshellcommand','shellcommand','run','python'}:
        cmd=_arg(args,'command','cmd','script','code')
        if name=='python' and 'code' in args and not args.get('command'):
            cmd=_file_edit_script('/tmp/salm_tool_exec.py',new=cmd)+'\npython3 /tmp/salm_tool_exec.py'
        return cmd or None
    if name in {'read','readfile','cat','view'}:
        path=_arg(args,'file_path','path','filename')
        if not path:return None
        return 'sed -n "1,320p" '+shlex.quote(path)
    if name in {'write','writefile','createfile'}:
        path=_arg(args,'file_path','path','filename')
        return _file_edit_script(path,new=_arg(args,'content','text')) if path else None
    if name in {'edit','editfile','replace'}:
        path=_arg(args,'file_path','path','filename')
        old=_arg(args,'old_string','old_text','old')
        new=_arg(args,'new_string','new_text','new')
        return _file_edit_script(path,old=old,new=new) if path and old else None
    if name in {'grep','search','glob'}:
        pattern=_arg(args,'pattern','query','glob')
        path=_arg(args,'path','directory') or '.'
        return ('find '+shlex.quote(path)+' -type f -name '+shlex.quote(pattern)+' | head -100' if name=='glob'
                else 'grep -R -n -- '+shlex.quote(pattern)+' '+shlex.quote(path)+' | head -150') if pattern else None
    return None


def forbidden_command(command: str) -> bool:
    """Refuse obvious direct inspection of Harbor grader/solution directories."""
    bad=(r'/logs/verifier',r'/tests/test\.sh',r'/tests/test_outputs',
         r'/solution/solve',r'/logs/oracle')
    return any(re.search(p,command,re.IGNORECASE) for p in bad)
