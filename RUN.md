# Running Prometheus on "Monte Cristo" (Windows desktop + WSL2)

This guide takes you from a fresh WSL2 Ubuntu shell to a live `prometheus run`
against **NVIDIA's hosted API** (the default backend), with an optional swap to a
**local NIM** if this desktop has a capable NVIDIA GPU.

> **Safety TL;DR.** The first run is safe by design: **dry-run is ON** (tool side
> effects are simulated), **authz is deny-by-default**, **budget caps are on**, and
> only two harmless in-process tools are registered. Nothing touches your real
> filesystem or network until you explicitly opt in. Read **§6** before you do.
>
> **Known limitation (from the code review):** the per-tool **execution sandbox is a
> stub**. Every tool call is gated by the authz engine + dual-LLM taint gate, but if
> you pass `--no-dry-run` with `--allow-fs`/`--allow-net`, tools run **unsandboxed in
> your WSL process**. Keep dry-run on unless you have read §6.

---

## 1. Prerequisites (one time)

In **Windows PowerShell** (only if WSL isn't installed yet):

```powershell
wsl --install -d Ubuntu
```

Then open **Ubuntu (WSL)** and confirm Python 3.10+:

```bash
python3 --version        # need >= 3.10
sudo apt-get update && sudo apt-get install -y python3-venv git   # if missing
```

## 2. Get the code

If you received the zip, unzip it in WSL (use the WSL filesystem, not /mnt/c, for speed):

```bash
mkdir -p ~/src && cd ~/src
unzip /mnt/c/Users/<you>/Downloads/prometheus-harness-rig.zip -d prometheus-harness
cd prometheus-harness
```

Or clone and switch to the rig branch:

```bash
cd ~/src
git clone https://github.com/IAMGODIAM/prometheus-harness.git
cd prometheus-harness
git checkout rig/nvidia-runnable
```

## 3. Python venv + install (torch is OPTIONAL)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Base install — enough to RUN the agent (NO torch, fast):
pip install -e .

# OPTIONAL: add the J-lens interpretability engine (heavy — torch/transformers):
# pip install -e '.[jlens]'
```

Sanity check the CLI and run the **offline** smoke (no key needed):

```bash
prometheus --help
prometheus run "Say hello via a safe tool." --provider mock
```

You should see `status: completion_claimed`, `verification: ACCEPT`, a gated `echo`
tool call, and a safety block. That proves the loop plans → acts → verifies → gates.

## 4. Configure the NVIDIA-hosted backend (default)

Get an API key from your NVIDIA account (build.nvidia.com) — it starts with `nvapi-`.

```bash
cp .env.example .env
# edit .env if you want a different model; then load it:
set -a && source .env && set +a

# put your real key in the shell (do NOT commit it):
export NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxx
```

Defaults already point at `https://integrate.api.nvidia.com/v1` with model
`meta/llama-3.1-8b-instruct`. Change `PROMETHEUS_LLM_MODEL` in `.env` to any model
your key can serve.

## 5. First live run (still safe: dry-run ON)

```bash
prometheus run "Summarize what a safe agent run looks like." --provider openai
```

This calls NVIDIA for planning/verification, but tool side effects are **simulated**
(dry-run). Add `--json` for the full result + trace.

## 6. Going live with real tools (opt-in, read §safety first)

Dry-run governs *tool execution*, not the LLM call. To let the agent actually use
tools:

```bash
# read-only filesystem tool, restricted to a root you choose:
prometheus run "Read ./notes.txt and summarize it" \
  --provider openai --no-dry-run --allow-fs --fs-root ./sandbox_dir

# network tool (adds an explicit authz permit; still budgeted + trifecta-gated):
prometheus run "Fetch example.com and summarize" \
  --provider openai --no-dry-run --allow-net
```

**What protects you even here:** deny-by-default authz (only classified/permitted
tools run), per-run budgets (`--max-tool-calls`, plus `prometheus.yaml` caps on
writes/external/irreversible), the dual-LLM Lethal-Trifecta gate (blocks
private-data + untrusted-content + exfiltration in one session), and `read_file`
path-restriction to `--fs-root`.

**What does NOT protect you:** there is no execution sandbox. `read_file`/`http_get`
run in your WSL process. Only enable them against inputs you trust, ideally inside a
throwaway directory. See RIG_NOTES.md.

## 7. Optional: local NIM instead of hosted (GPU-gated)

Monte Cristo is a desktop, so it may have a capable NVIDIA GPU. Check inside WSL:

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

- **Command errors / not found** → no GPU passthrough to WSL; **stay on the hosted
  backend** (§4). (Enable the WSL CUDA driver on Windows if you want GPU passthrough.)
- **Prints a GPU with enough VRAM** (rule of thumb: ~16 GB+ for an 8B NIM, more for
  larger) → you can run a local NIM. Start one (example):

```bash
# requires NVIDIA NGC access + Docker with GPU support in WSL:
docker run --rm --gpus all -p 8000:8000 \
  nvcr.io/nim/meta/llama-3.1-8b-instruct:latest
```

Then swap the backend with **one env var** (no code change) and run:

```bash
export PROMETHEUS_LLM_BASE_URL=http://localhost:8000/v1
prometheus run "Say hello via a safe tool." --provider openai
```

A local NIM needs no `NVIDIA_API_KEY` for inference; the OpenAI-compatible client
sends the header but the NIM ignores auth. Swap back by unsetting/resetting
`PROMETHEUS_LLM_BASE_URL`.

## 8. Troubleshooting

- `ERROR: No API key found` → `export NVIDIA_API_KEY=nvapi-...` (or use `--provider mock`).
- `command not found: prometheus` → activate the venv (`source .venv/bin/activate`) or run `python -m prometheus.cli ...`.
- Model 404 / not authorized → set `PROMETHEUS_LLM_MODEL` to a model your key can access.
- Want the interpretability engine (`prometheus fit/probe`)? Install the extra: `pip install -e '.[jlens]'`.
- Verify the build: `pip install pytest pytest-asyncio && python -m pytest tests/test_harness.py tests/test_authz_memory.py tests/test_run_smoke.py -q`
