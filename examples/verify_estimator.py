"""Brute-force correctness check for the J-lens Jacobian estimator.

Fits the batched-VJP estimator (prometheus.jlens.fitting.jacobian_for_prompt)
on a tiny random-init GPT-2 and compares every layer's Jacobian against a
reference computed with torch.autograd.functional.jacobian (full Jacobian of
the summed valid-target final residual w.r.t. the source-layer activation,
averaged over valid source positions).

Expected output: max_abs_diff == 0 at every layer, and the target layer's
Jacobian equal to the identity (norm == sqrt(d_model)).

Run:  python examples/verify_estimator.py
Deps: torch (CPU is fine), transformers.
"""

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from prometheus.jlens.fitting import jacobian_for_prompt

D_MODEL = 32
SEQ_LEN = 12
SKIP = 2
SRC_LAYERS = [0, 1, 2]
TARGET = 2


def main() -> None:
    torch.manual_seed(0)
    cfg = GPT2Config(n_embd=D_MODEL, n_layer=3, n_head=4, vocab_size=200, n_positions=64)
    model = GPT2LMHeadModel(cfg).eval()
    input_ids = torch.randint(10, 200, (1, SEQ_LEN))

    j_est = jacobian_for_prompt(
        model, input_ids, SRC_LAYERS, TARGET,
        d_model=D_MODEL, dim_batch=8, skip_first_n=SKIP, device="cpu",
    )

    layers = model.transformer.h

    def reference_jacobian(src_layer: int) -> torch.Tensor:
        cap = {}

        def cap_hook(m, i, o):
            cap["a"] = (o[0] if isinstance(o, tuple) else o).detach()

        handle = layers[src_layer].register_forward_hook(cap_hook)
        with torch.no_grad():
            model(input_ids)
        handle.remove()
        clean_act = cap["a"][0]  # (seq, d)

        def func(x: torch.Tensor) -> torch.Tensor:
            holder = {}

            def repl_hook(m, i, o):
                if isinstance(o, tuple):
                    return (x.unsqueeze(0),) + o[1:]
                return x.unsqueeze(0)

            def tgt_hook(m, i, o):
                holder["t"] = o[0] if isinstance(o, tuple) else o

            h1 = layers[src_layer].register_forward_hook(repl_hook)
            h2 = (
                layers[TARGET].register_forward_hook(tgt_hook)
                if TARGET != src_layer
                else None
            )
            model(input_ids)
            h1.remove()
            if h2:
                h2.remove()
            t = holder["t"] if TARGET != src_layer else x.unsqueeze(0)
            return t[0, SKIP:SEQ_LEN, :].sum(dim=0)  # (d,)

        jac = torch.autograd.functional.jacobian(func, clean_act)  # (d_out, seq, d_in)
        return jac[:, SKIP : SEQ_LEN - 1, :].mean(dim=1)

    for layer in SRC_LAYERS:
        j_ref = reference_jacobian(layer)
        diff = (j_est[layer] - j_ref).abs().max().item()
        print(
            f"layer {layer}: est_norm={j_est[layer].norm():.4f} "
            f"ref_norm={j_ref.norm():.4f} max_abs_diff={diff:.2e}"
        )
        assert diff < 1e-4, f"MISMATCH at layer {layer}"

    print("ESTIMATOR EXACT ✓")


if __name__ == "__main__":
    main()
