"""Which wire schema makes a small model emit usable tool calls?

Compares an all-required flat schema with the discriminated-union `Step` schema by sampling
the same agent prompt (calc.py already read, next step should be an edit) six times each.
Needs a running Ollama with qwen2.5-coder:1.5b. Results are in 2026-10-16-local-model-results.md;
those were recorded with an earlier wording of the system prompt, so exact outputs will differ.
"""
import json, sys, httpx
from typing import Annotated, Literal, Union
from pydantic import BaseModel, Field
from wombat_harness.agent.loop import SYSTEM

ROOT = "/tmp/proj"
sysmsg = SYSTEM.replace("<ROOT>", ROOT).replace("<LISTING>", "calc.py\ntest_calc.py")
msgs = [
 {"role":"system","content":sysmsg},
 {"role":"user","content":"The test in test_calc.py is failing. Find the bug in calc.py and fix it, then run the tests."},
 {"role":"assistant","content":'{"thought":"read calc.py","tool":"read_file","path":"calc.py"}'},
 {"role":"user","content":"Tool result (read_file):\n   1  def add(a, b):\n   2      return a - b\n   3  \n   4  \n   5  def mul(a, b):\n   6      return a * b"},
]

class ReqFlat(BaseModel):
    thought: str
    tool: Literal["list_dir","read_file","grep","write_file","edit_file","bash","finish"]
    path: str; content: str; old: str; new: str; command: str; pattern: str; message: str

class Edit(BaseModel):
    tool: Literal["edit_file"]; path: str = Field(min_length=1); old: str = Field(min_length=1); new: str
class Read(BaseModel):
    tool: Literal["read_file"]; path: str = Field(min_length=1)
class Bash(BaseModel):
    tool: Literal["bash"]; command: str = Field(min_length=1)
class Fin(BaseModel):
    tool: Literal["finish"]; message: str
class Union_(BaseModel):
    thought: str
    action: Annotated[Union[Edit, Read, Bash, Fin], Field(discriminator="tool")]

def sample(schema, n=6):
    out = []
    for _ in range(n):
        body = {"model":"qwen2.5-coder:1.5b","messages":msgs,"temperature":0.2,
                "response_format":{"type":"json_schema","json_schema":{"name":"S","schema":schema.model_json_schema(),"strict":True}}}
        try:
            r = httpx.post("http://localhost:11434/v1/chat/completions", json=body, timeout=180).json()
            out.append(r["choices"][0]["message"]["content"].replace("\n","")[:230])
        except Exception as e:
            out.append(f"ERR {e}")
    return out

for name, sc in (("required-flat", ReqFlat), ("union", Union_)):
    print("==", name)
    for o in sample(sc): print("  ", o)
