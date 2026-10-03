"""One adapter per generator. Each returns a `Design`, and each records what it
had to do to get there.

All four sources speak LDraw, which is why it is the wire format:

  BrickGPT   a 1-unit-tall voxel grid; `Brick.to_ldr` converts, and this adapter
             reimplements that conversion so the text format can be read without
             importing BrickGPT and its Llama dependency
  BrickNet   a connector graph; `graph_to_ldr` realises absolute poses
  LegoACE    `.ldr` natively
  reference  official or human `.ldr`, the control arm

An adapter is allowed to repair, never to hide. Everything it changed appears in
`Design.ingest` and travels with the score.
"""
from ppbench.baselines.adapters.brickgpt import from_brickgpt_txt, from_brickgpt_json
from ppbench.baselines.adapters.bricknet import from_bricknet_npz, n_graphs
from ppbench.baselines.adapters.ldr import from_ldr_file

__all__ = ["from_brickgpt_txt", "from_brickgpt_json", "from_bricknet_npz",
           "n_graphs", "from_ldr_file"]
