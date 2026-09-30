#!/usr/bin/env python3
"""
Warm Pool Metrics Endpoint - Simple HTTP server for exposing pool metrics.

Run this to expose /metrics endpoint for Prometheus scraping or JSON inspection.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, "/home/user/src/prom/repo")

from prometheus.harness.subagent_pool import get_pool_metrics
from prometheus.harness.model_client import SubagentTier, EffortLevel, ProviderFactory, MockProvider
from prometheus.harness.tools import build_default_toolset
from prometheus.authz.engine import AuthzEngine
from prometheus.harness.jlens_gate import load_jlens_gate

try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore

logger = logging.getLogger(__name__)


async def setup_pool():
    """Initialize the global pool for metrics."""
    provider_factory = ProviderFactory()
    tool_router = build_default_toolset(dry_run=True)
    authz_engine = AuthzEngine()
    jlens_gate = load_jlens_gate(enable=False)

    from prometheus.harness.subagent_pool import init_global_pool
    init_global_pool(
        max_size=20,
        provider_factory=provider_factory,
        tool_router=tool_router,
        authz_engine=authz_engine,
        jlens_gate=jlens_gate,
    )
    logger.info("Global pool initialized for metrics endpoint")


async def metrics_handler(request):
    """Handle /metrics endpoint - returns JSON metrics."""
    metrics = get_pool_metrics()
    return web.json_response(metrics)


async def health_handler(request):
    """Handle /health endpoint."""
    return web.json_response({"status": "healthy", "service": "warm-pool-metrics"})


async def prometheus_handler(request):
    """Handle /metrics endpoint in Prometheus format."""
    metrics = get_pool_metrics()
    
    lines = [
        "# HELP warm_pool_size Current number of warm subagents in pool",
        "# TYPE warm_pool_size gauge",
        f"warm_pool_size {metrics['pool_size']}",
        "",
        "# HELP warm_pool_max_size Maximum pool size",
        "# TYPE warm_pool_max_size gauge",
        f"warm_pool_max_size {metrics['pool_max_size']}",
        "",
        "# HELP warm_pool_utilization Pool utilization ratio (0-1)",
        "# TYPE warm_pool_utilization gauge",
        f"warm_pool_utilization {metrics['pool_utilization']:.4f}",
        "",
        "# HELP warm_pool_total_requests Total pool requests",
        "# TYPE warm_pool_total_requests counter",
        f"warm_pool_total_requests {metrics['total_requests']}",
        "",
        "# HELP warm_pool_cache_hits Total cache hits",
        "# TYPE warm_pool_cache_hits counter",
        f"warm_pool_cache_hits {metrics['cache_hits']}",
        "",
        "# HELP warm_pool_cache_misses Total cache misses",
        "# TYPE warm_pool_cache_misses counter",
        f"warm_pool_cache_misses {metrics['cache_misses']}",
        "",
        "# HELP warm_pool_hit_rate Cache hit rate (0-1)",
        "# TYPE warm_pool_hit_rate gauge",
        f"warm_pool_hit_rate {metrics['hit_rate']:.4f}",
        "",
        "# HELP warm_pool_creations Total subagent creations",
        "# TYPE warm_pool_creations counter",
        f"warm_pool_creations {metrics['creations']}",
        "",
        "# HELP warm_pool_evictions Total evictions",
        "# TYPE warm_pool_evictions counter",
        f"warm_pool_evictions {metrics['evictions']}",
        "",
        "# HELP warm_pool_expired Total expired entries",
        "# TYPE warm_pool_expired counter",
        f"warm_pool_expired {metrics['expired']}",
        "",
    ]
    
    # Tier distribution
    for tier, count in metrics.get("tier_distribution", {}).items():
        lines.append(f'# HELP warm_pool_tier_subagents Number of subagents by tier')
        lines.append(f'# TYPE warm_pool_tier_subagents gauge')
        lines.append(f'warm_pool_tier_subagents{{tier="{tier}"}} {count}')
        lines.append("")
    
    # Effort distribution
    for effort, count in metrics.get("effort_distribution", {}).items():
        lines.append(f'# HELP warm_pool_effort_subagents Number of subagents by effort')
        lines.append(f'# TYPE warm_pool_effort_subagents gauge')
        lines.append(f'warm_pool_effort_subagents{{effort="{effort}"}} {count}')
        lines.append("")
    
    # Specialization distribution
    for spec, count in metrics.get("specialization_distribution", {}).items():
        lines.append(f'# HELP warm_pool_specialization_subagents Number of subagents by specialization')
        lines.append(f'# TYPE warm_pool_specialization_subagents gauge')
        lines.append(f'warm_pool_specialization_subagents{{specialization="{spec}"}} {count}')
        lines.append("")
    
    return web.Response(text="\n".join(lines), content_type="text/plain")


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if not AIOHTTP_AVAILABLE:
        logger.error("aiohttp not available. Install with: pip install aiohttp")
        print("Install aiohttp to run the metrics server: pip install aiohttp")
        # Just print metrics once
        await setup_pool()
        metrics = get_pool_metrics()
        print(json.dumps(metrics, indent=2))
        return

    await setup_pool()

    app = web.Application()
    app.router.add_get("/metrics", metrics_handler)
    app.router.add_get("/metrics/prometheus", prometheus_handler)
    app.router.add_get("/health", health_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, "0.0.0.0", 8422)
    await site.start()
    
    logger.info("Warm Pool Metrics server started on http://0.0.0.0:8422")
    logger.info("  JSON: http://localhost:8422/metrics")
    logger.info("  Prometheus: http://localhost:8422/metrics/prometheus")
    logger.info("  Health: http://localhost:8422/health")
    
    # Keep running
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())