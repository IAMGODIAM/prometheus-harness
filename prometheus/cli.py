"""CLI Interface — Command-line interface for Prometheus operations.

Commands:
- prometheus fit: Fit a J-lens on a model
- prometheus probe: Run J-lens probe on a prompt
- prometheus watchlist: Score a prompt against watchlists
- prometheus decompose: J-space decomposition
- prometheus serve: Start MCP server (stdio or HTTP)
- prometheus deploy: Deploy to Cloudflare Workers
- prometheus config: Manage configuration
- prometheus status: Show system status
- prometheus run: Run an agent task end-to-end — plan/act/verify/gate (RIG)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from prometheus.config import HarnessConfig


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="prometheus",
        description="Prometheus — MCP-First J-Lens Anchored Agentic Harness",
    )
    parser.add_argument("--config", "-c", type=str, default="prometheus.yaml", help="Config file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # fit command
    fit_parser = subparsers.add_parser("fit", help="Fit a J-lens on a model")
    fit_parser.add_argument("--model", "-m", type=str, help="Model ID or path")
    fit_parser.add_argument("--samples", "-n", type=int, default=512, help="Number of samples")
    fit_parser.add_argument("--batch-size", "-b", type=int, default=8, help="Batch size")
    fit_parser.add_argument("--output", "-o", type=str, default=None, help="Output path for lens artifact")
    fit_parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda, mps)")
    fit_parser.add_argument("--dtype", type=str, default="float32", help="Data type")

    # probe command
    probe_parser = subparsers.add_parser("probe", help="Run J-lens probe on a prompt")
    probe_parser.add_argument("prompt", type=str, help="Text prompt to probe")
    probe_parser.add_argument("--model", "-m", type=str, help="Model ID")
    probe_parser.add_argument("--lens", "-l", type=str, help="Lens artifact path")
    probe_parser.add_argument("--layers", type=str, default=None, help="Layers to probe (e.g., '8,9,10')")
    probe_parser.add_argument("--top-k", "-k", type=int, default=10, help="Top-k tokens")
    probe_parser.add_argument("--json", action="store_true", help="JSON output")

    # watchlist command
    wl_parser = subparsers.add_parser("watchlist", help="Score prompt against watchlists")
    wl_parser.add_argument("prompt", type=str, help="Text prompt to score")
    wl_parser.add_argument("--model", "-m", type=str, help="Model ID")
    wl_parser.add_argument("--lens", "-l", type=str, help="Lens artifact path")
    wl_parser.add_argument("--categories", type=str, default=None, help="Categories (comma-separated)")
    wl_parser.add_argument("--json", action="store_true", help="JSON output")

    # decompose command
    dec_parser = subparsers.add_parser("decompose", help="J-space decomposition")
    dec_parser.add_argument("prompt", type=str, help="Text prompt")
    dec_parser.add_argument("--layer", type=int, required=True, help="Layer index")
    dec_parser.add_argument("--position", type=int, required=True, help="Token position")
    dec_parser.add_argument("--k", type=int, default=25, help="Sparsity level")
    dec_parser.add_argument("--model", "-m", type=str, help="Model ID")
    dec_parser.add_argument("--lens", "-l", type=str, help="Lens artifact path")

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start MCP server")
    serve_parser.add_argument("--transport", "-t", choices=["stdio", "http"], default="stdio")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="HTTP host")
    serve_parser.add_argument("--port", type=int, default=8080, help="HTTP port")
    serve_parser.add_argument("--model", "-m", type=str, help="Model to load")
    serve_parser.add_argument("--lens", "-l", type=str, help="Lens artifact to load")

    # deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy to Cloudflare Workers")
    deploy_parser.add_argument("--name", type=str, default="prometheus-mcp", help="Worker name")
    deploy_parser.add_argument("--upstream", type=str, help="Upstream Prometheus URL")
    deploy_parser.add_argument("--output-dir", "-o", type=str, default="./worker", help="Output directory")
    deploy_parser.add_argument("--deploy", action="store_true", help="Actually deploy (vs just generate)")

    # config command
    config_parser = subparsers.add_parser("config", help="Manage configuration")
    config_parser.add_argument("action", choices=["init", "show", "validate"], help="Config action")

    # status command
    subparsers.add_parser("status", help="Show system status")

    # run command (RIG) — run an agent task end-to-end through the orchestrator loop
    run_parser = subparsers.add_parser(
        "run", help="Run an agent task end-to-end (plan -> act -> verify -> gate)"
    )
    run_parser.add_argument("task", type=str, help="The task/prompt for the agent")
    run_parser.add_argument(
        "--provider", choices=["openai", "mock"], default=None,
        help="LLM provider. Default: env PROMETHEUS_LLM_PROVIDER, else 'openai' (NVIDIA hosted).",
    )
    run_parser.add_argument(
        "--dry-run", dest="dry_run", action="store_true", default=True,
        help="Simulate tool side effects — DEFAULT ON (safe first-run posture).",
    )
    run_parser.add_argument(
        "--no-dry-run", dest="dry_run", action="store_false",
        help="Allow real tool execution (UNSANDBOXED — see RUN.md before using).",
    )
    run_parser.add_argument("--allow-fs", action="store_true", help="Register read-only filesystem tool (opt-in).")
    run_parser.add_argument("--allow-net", action="store_true", help="Register network tool + explicit permit (opt-in).")
    run_parser.add_argument("--fs-root", type=str, default=None, help="Root dir for read_file (default: cwd).")
    run_parser.add_argument("--enable-jlens-gate", action="store_true", help="Enable J-lens gate (needs torch+fitted lens; else no-op).")
    run_parser.add_argument("--max-tool-calls", type=int, default=None, help="Override the tool-call budget.")
    run_parser.add_argument("--json", action="store_true", help="JSON output.")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Load config
    config = HarnessConfig.from_yaml(args.config)

    # Dispatch command
    commands = {
        "fit": cmd_fit,
        "probe": cmd_probe,
        "watchlist": cmd_watchlist,
        "decompose": cmd_decompose,
        "serve": cmd_serve,
        "deploy": cmd_deploy,
        "config": cmd_config,
        "status": cmd_status,
        "run": cmd_run,
    }

    cmd_fn = commands.get(args.command)
    if cmd_fn:
        asyncio.run(cmd_fn(args, config))
    else:
        parser.print_help()


async def cmd_fit(args: argparse.Namespace, config: HarnessConfig) -> None:
    """Fit a J-lens on a model."""
    from prometheus.jlens.model_adapter import ModelAdapter
    from prometheus.jlens.fitting import JLensFitter

    model_id = args.model or config.model.model_id
    print(f"Fitting J-lens on model: {model_id}")
    print(f"  Samples: {args.samples}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Device: {args.device}")

    adapter = ModelAdapter.from_pretrained(
        model_id,
        device=args.device,
        dtype=_parse_dtype(args.dtype),
    )

    fitter = JLensFitter(
        adapter=adapter,
        n_samples=args.samples,
        batch_size=args.batch_size,
    )

    lens = await fitter.fit()

    output_path = args.output or f".prometheus/cache/{model_id.replace('/', '_')}_jlens.pt"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    lens.save(output_path)
    print(f"Lens saved to: {output_path}")


async def cmd_probe(args: argparse.Namespace, config: HarnessConfig) -> None:
    """Run J-lens probe on a prompt."""
    from prometheus.jlens.model_adapter import ModelAdapter
    from prometheus.jlens.lens import JacobianLens

    model_id = args.model or config.model.model_id
    adapter = ModelAdapter.from_pretrained(model_id, device=config.model.device)

    lens_path = args.lens or f".prometheus/cache/{model_id.replace('/', '_')}_jlens.pt"
    lens = JacobianLens.load(lens_path)

    layers = None
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]

    tokens = adapter.tokenize(args.prompt)
    input_ids = tokens["input_ids"]

    if layers is None:
        n = adapter.n_layers
        layers = list(range(n // 3, 2 * n // 3))

    activations = adapter.get_activations(input_ids, layers=layers)

    results = []
    for layer_idx in layers:
        if layer_idx not in activations:
            continue
        act = activations[layer_idx][0]
        for pos in range(act.shape[0]):
            h = act[pos]
            logits = lens.apply(
                h.unsqueeze(0), layer_idx,
                norm_fn=adapter.final_norm,
                unembed=adapter.unembed_weight,
            )
            values, indices = logits.squeeze().topk(args.top_k)
            top_tokens = [
                {"token": adapter.decode_tokens([idx.item()])[0], "score": val.item()}
                for val, idx in zip(values, indices)
            ]
            results.append({
                "layer": layer_idx,
                "position": pos,
                "input_token": adapter.decode_tokens([input_ids[0, pos].item()])[0],
                "top_tokens": top_tokens,
            })

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            print(f"\nLayer {r['layer']}, Position {r['position']} ('{r['input_token']}'):")
            for t in r["top_tokens"][:5]:
                print(f"  {t['token']:>15s}  {t['score']:.4f}")


async def cmd_watchlist(args: argparse.Namespace, config: HarnessConfig) -> None:
    """Score prompt against watchlists."""
    print(f"Scoring: '{args.prompt[:50]}...'")
    print("(Requires fitted lens — use 'prometheus fit' first)")


async def cmd_decompose(args: argparse.Namespace, config: HarnessConfig) -> None:
    """J-space decomposition."""
    print(f"Decomposing layer={args.layer}, position={args.position}, k={args.k}")
    print("(Requires fitted lens — use 'prometheus fit' first)")


async def cmd_serve(args: argparse.Namespace, config: HarnessConfig) -> None:
    """Start MCP server."""
    from prometheus.mcp_server import create_jlens_server

    print(f"Starting Prometheus MCP server ({args.transport} transport)")

    server = create_jlens_server(enable_steering=config.jlens.enable_steering)

    if args.transport == "stdio":
        print("Listening on stdio...")
        # In production, this would run the FastMCP stdio transport
        try:
            import fastmcp
            await server.run(transport="stdio")
        except ImportError:
            print("FastMCP not installed. Install with: pip install fastmcp")
    else:
        print(f"Listening on http://{args.host}:{args.port}/mcp")
        try:
            import fastmcp
            await server.run(transport="streamable-http", host=args.host, port=args.port)
        except ImportError:
            print("FastMCP not installed. Install with: pip install fastmcp")


async def cmd_deploy(args: argparse.Namespace, config: HarnessConfig) -> None:
    """Deploy to Cloudflare Workers."""
    from prometheus.mcp_server.transport import CloudflareWorkersTransport

    transport = CloudflareWorkersTransport(
        worker_name=args.name,
        upstream_url=args.upstream,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "src").mkdir(exist_ok=True)

    # Generate worker script
    worker_script = transport.generate_worker_script()
    (output_dir / "src" / "index.js").write_text(worker_script)

    # Generate wrangler config
    wrangler_config = transport.generate_wrangler_config()
    (output_dir / "wrangler.toml").write_text(wrangler_config)

    # Generate package.json
    package_json = json.dumps({
        "name": args.name,
        "version": "0.1.0",
        "private": True,
        "scripts": {
            "dev": "wrangler dev",
            "deploy": "wrangler deploy",
        },
        "devDependencies": {
            "wrangler": "^3.0.0",
        },
    }, indent=2)
    (output_dir / "package.json").write_text(package_json)

    print(f"Worker generated in: {output_dir}")
    print(f"  Worker name: {args.name}")
    print(f"  Upstream: {args.upstream or '(edge-only mode)'}")
    print(f"\nTo deploy:")
    print(f"  cd {output_dir}")
    print(f"  npm install")
    print(f"  npx wrangler deploy")

    if args.deploy:
        import subprocess
        subprocess.run(["npm", "install"], cwd=output_dir)
        subprocess.run(["npx", "wrangler", "deploy"], cwd=output_dir)


async def cmd_config(args: argparse.Namespace, config: HarnessConfig) -> None:
    """Manage configuration."""
    if args.action == "init":
        config.save("prometheus.yaml")
        print("Created prometheus.yaml with default configuration")
    elif args.action == "show":
        print(config.to_yaml())
    elif args.action == "validate":
        print("Configuration valid.")
        print(f"  Model: {config.model.model_id}")
        print(f"  Device: {config.model.device}")
        print(f"  J-lens samples: {config.jlens.n_samples}")
        print(f"  Steering enabled: {config.jlens.enable_steering}")
        print(f"  Watchlist categories: {len(config.watchlist.categories)}")


async def cmd_status(args: argparse.Namespace, config: HarnessConfig) -> None:
    """Show system status."""
    import torch

    print("Prometheus System Status")
    print("=" * 40)
    print(f"  Config: {args.config}")
    print(f"  Model: {config.model.model_id}")
    print(f"  Device: {config.model.device}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"  MPS available: {torch.backends.mps.is_available()}")
    print(f"  Steering: {'enabled' if config.jlens.enable_steering else 'disabled'}")
    print(f"  Vault path: {config.memory.vault_path}")


async def cmd_run(args: argparse.Namespace, config: HarnessConfig) -> None:
    """Run an agent task end-to-end through the orchestrator (RIG)."""
    import os
    from prometheus.harness.runner import run_task, RunConfig
    from prometheus.harness.model_client import (
        MockProvider,
        OpenAICompatibleProvider,
        build_provider_from_env,
        NVIDIA_DEFAULT_BASE_URL,
        DEFAULT_MODEL_ID,
    )

    if args.provider == "mock":
        provider = MockProvider()
    elif args.provider == "openai":
        provider = OpenAICompatibleProvider(
            base_url=os.environ.get("PROMETHEUS_LLM_BASE_URL", NVIDIA_DEFAULT_BASE_URL),
            model_id=os.environ.get("PROMETHEUS_LLM_MODEL", DEFAULT_MODEL_ID),
            api_key_env=os.environ.get("PROMETHEUS_API_KEY_ENV", "NVIDIA_API_KEY"),
        )
    else:
        provider = build_provider_from_env()

    rc = RunConfig(
        dry_run=args.dry_run,
        allow_fs=args.allow_fs,
        allow_net=args.allow_net,
        enable_jlens_gate=args.enable_jlens_gate,
        max_tool_calls=args.max_tool_calls,
        fs_root=args.fs_root,
    )

    print(
        f"[prometheus run] provider={getattr(provider, 'model_id', '?')} "
        f"dry_run={rc.dry_run} allow_fs={rc.allow_fs} allow_net={rc.allow_net} "
        f"jlens_gate={'on' if rc.enable_jlens_gate else 'off'}"
    )
    if not rc.dry_run:
        print(
            "  WARNING: --no-dry-run set. Tools may perform REAL actions and are NOT "
            "sandboxed (exec-sandbox stage is a stub). See RUN.md."
        )

    try:
        result = await run_task(args.task, config=config, provider=provider, run=rc)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(2)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return

    print("\n=== RESULT ===")
    print(f"status:       {result.get('status')}")
    verification = result.get("verification")
    if verification:
        print(f"verification: {verification.get('status')} (score {verification.get('score')})")
    print(f"tool_calls:   {result.get('tool_calls')}")
    print(f"safety:       {result.get('safety')}")
    print(f"trace spans:  {result.get('trace', {}).get('span_count')}")


def _parse_dtype(dtype_str: str):
    """Parse dtype string to torch dtype."""
    import torch
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str, torch.float32)


if __name__ == "__main__":
    main()
