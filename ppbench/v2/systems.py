"""The 30 systems reported in the paper's 200-task tables (Tier B agents and domain generators)."""
CLOSED = ("gpt-6-astra", "gpt-5.6-sol", "claude-opus-5", "claude-fable-5.1", "claude-sonnet-5", "claude-haiku-4.5")
OPEN = ("qwen3.5-27b", "qwen3.5-35b-a3b", "gpt-oss-120b", "gemma-4-31b-it", "qwen3-vl-32b-instruct",
        "qwen3-vl-30b-a3b-instruct", "qwen3-vl-8b-instruct", "qwen3-4b-instruct-2507", "internvl3.5-38b",
        "internvl3.5-8b", "minicpm-v-4.5", "ernie-4.5-vl-28b-a3b", "ministral-3-8b-instruct-2512")
EXT = ("partcrafter_np15", "particulate-partcrafter", "partcrafter_np8", "particulate-partcrafter-np8", "partpacker",
       "particulate-partpacker", "physx-anything", "cubepart_core", "particulate-cube3d", "brickgpt", "legoace")
AGENTS = CLOSED + OPEN
