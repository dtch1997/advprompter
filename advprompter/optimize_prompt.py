import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import transformers
import torch
import json
import torch.nn.functional as F
import functools
import gradio
from IPython.display import HTML, display
from typing import Any

from sae_lens.toolkit.pretrained_saes import get_gpt2_res_jb_saes
from dreamy.epo import epo, add_fwd_hooks, build_pareto_frontier
from dreamy.attribution import resample_viz


def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return device

@functools.lru_cache(1)
def get_model_and_tokenizer():
    # Load the model
    model_name = "gpt2"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        torch_dtype="auto",
        use_cache=False,
        device_map=get_device()
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

@functools.lru_cache(20)
def get_sae(layer: int):
    hook_point = f"blocks.{layer}.hook_resid_pre"
    # if the SAEs were stored with precomputed feature sparsities,
    #  those will be return in a dictionary as well.
    saes, _ = get_gpt2_res_jb_saes(hook_point)
    sparse_autoencoder = saes[hook_point]
    sparse_autoencoder.to(get_device())
    sparse_autoencoder.cfg.device = get_device()

    return sparse_autoencoder

def smin(xs, dim=-1, scale = 0.01, offset=10):
    if xs.shape[-1] == 1:
        return xs[..., 0]
    assert dim == -1
    # relu and norm
    return F.relu(xs + offset).norm(dim=dim, p=0.1) * scale

def neuron_runner(
    layer: int,
    neurons: list[int]
):

    model, tokenizer = get_model_and_tokenizer()
    sparse_autoencoder = get_sae(layer)

    def f(*model_args, **model_kwargs):
        out = {}

        def get_target(module, input, output):
            hidden_pre = sparse_autoencoder._encode_with_hidden_pre(
                input[0])[1]
            
            all_acts = hidden_pre[..., neurons][:, -1:]
            # positions, indices = zip(*neurons)
            # all_acts = hidden_pre[..., positions, indices]
            # print(all_acts.shape)
            # TODO: replace min with soft min? 
            out["target"] = smin(all_acts, dim = -1, scale = 0.001).mean(dim=-1)

        with add_fwd_hooks(
            [
                (model.transformer.h[layer], get_target),
            ]
        ):
            if "inputs_embeds" in model_kwargs:
                # Add the BOS embedding
                model_kwargs["inputs_embeds"] = torch.cat(
                    (
                        model.transformer.wte.weight[
                            [0] * len(model_kwargs["inputs_embeds"])
                        ].unsqueeze(1),
                        model_kwargs["inputs_embeds"],
                    ),
                    dim=1,
                )
            else:
                print(model_kwargs["input_ids"][0, :1], tokenizer.bos_token_id)
                assert tokenizer.bos_token_id is not None
                model_kwargs["input_ids"] = torch.cat(
                    (
                        torch.full(
                            (len(model_kwargs["input_ids"]), 1),
                            tokenizer.bos_token_id, 
                            device=model_kwargs["input_ids"].device,
                            dtype=model_kwargs["input_ids"].dtype,
                        ), # type: ignore
                        model_kwargs["input_ids"],
                    ),
                    dim=1,
                )
            out["logits"] = model(*model_args, **model_kwargs).logits
            out["logits"] = out["logits"][:, 1:]
        return out

    return f, model, tokenizer

def optimize_prompt(
    layer: int, 
    neurons: str,
    iters: int
) -> tuple[Any, str]:
    """ Return:
    - a string that optimizes the metric 
    - pareto plot of x-entropy vs target
    """
    neurons = json.loads(neurons)
    if isinstance(neurons, int):
        neurons = [neurons]
    neurons: list[int] = neurons

    runner, model, tokenizer = neuron_runner(
        layer=layer, neurons = neurons
    )
    history = epo(
        runner, model,
        tokenizer,  # type: ignore
        seed=1001,
        iters = iters,
        restart_xentropy = 0.5, 
        restart_xentropy_max_mult = 3.0,
        x_penalty_max=100,
        x_penalty_min=0.1,
    )

    token_strs: list[str] = tokenizer.batch_decode(history.ids[-1])
    x_ents: list[float] = history.xentropy[-1].tolist()
    targets: list[float] = history.target[-1].tolist()

    pareto = build_pareto_frontier(tokenizer, history)
    ordering = np.argsort(pareto.xentropy)

    fig, ax = plt.subplots()
    ax.scatter(pareto.xentropy, pareto.target, c='k', label='Pareto frontier')
    for i, k in enumerate(ordering):
        ax.text(
            pareto.xentropy[k] + 0.05, 
            pareto.target[k] + 0.05, 
            pareto.text[k], 
            fontsize=8, 
            rotation=-8, 
            va='top', 
            color='black', 
            alpha=1.0
        )

    return fig, "\n".join(pareto.text), "\n".join(token_strs)

demo = gradio.Interface(
    optimize_prompt,
    inputs=[
        gradio.Number(label="layer", minimum=0, maximum=12, value=10),
        gradio.Textbox(label="neurons", value="17165"),
        gradio.Slider(10, 1000, label="iters", step=10, value=1000)
    ],
    outputs=[
        gradio.Plot(label="Pareto Frontier Plot"),
        gradio.TextArea(label="Pareto-optimal Text"),
        gradio.TextArea(label="All text"),
    ]
)

demo.launch(share=True)


