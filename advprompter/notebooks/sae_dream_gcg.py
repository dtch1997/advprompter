# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import transformers
import torch
from IPython.display import HTML, display

from dreamy.epo import epo, add_fwd_hooks, build_pareto_frontier
from dreamy.attribution import resample_viz

np.set_printoptions(edgeitems=10, linewidth=100)
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", 500)
# %%
model_name = "gpt2"
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    torch_dtype="auto",
    use_cache=False,
    device_map="cuda"
)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

# %%
model.transformer.h[5]

# %%
# from sae_lens.training.session_loader import LMSparseAutoencoderSessionloader
from sae_lens.toolkit.pretrained_saes import get_gpt2_res_jb_saes

layer: int = 10
device = "cuda"
hook_point = f"blocks.{layer}.hook_resid_pre"

# if the SAEs were stored with precomputed feature sparsities,
#  those will be return in a dictionary as well.
saes, sparsities = get_gpt2_res_jb_saes(hook_point)
print(saes.keys())
sparse_autoencoder = saes[hook_point]
sparse_autoencoder.to(device)
sparse_autoencoder.cfg.device = device

# %%

import torch.nn.functional as F

# https://iquilezles.org/articles/smin/
def smin(xs, dim=-1, scale = 0.01):
    if xs.shape[-1] == 1:
        return xs[..., 0]
    assert dim == -1
    # relu and norm
    return F.relu(xs + 10).norm(dim=dim, p=0.1) * scale
    return xs.min(dim=dim).values
    k = 0.5
    assert xs.shape[-1] == 2
    h = torch.nn.functional.relu( k - torch.abs(xs[..., 0] - xs[..., 1])) / k
    return torch.minimum(xs[..., 0], xs[..., 1]) - h*h*k*(1.0/4.0)
    # k = 1.0/(1.0-(0.5**0.5));
    # a, b = xs[..., 0], xs[..., 1]
    # return torch.maximum(k, torch.minimum(a,b)) -
            # length(max(k-vec2(a,b),0.0));

# %% 

neurons = (234, 17165)

def neuron_runner(layer: int,
                  neurons: list[int]
                #   neurons: list[tuple[int, int]]
                  ):
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
                model_kwargs["input_ids"] = torch.cat(
                    (
                        torch.full(
                            (len(model_kwargs["input_ids"]), 1),
                            tokenizer.bos_token_id,
                            device=model_kwargs["input_ids"].device,
                            dtype=model_kwargs["input_ids"].dtype,
                        ),
                        model_kwargs["input_ids"],
                    ),
                    dim=1,
                )
            out["logits"] = model(*model_args, **model_kwargs).logits
            out["logits"] = out["logits"][:, 1:]
        return out

    return f
runner = neuron_runner(layer=layer, neurons = neurons)
history = epo(
    runner, model, tokenizer, 
    seed=1001,
    iters = 10, # 1000,
    restart_xentropy = 0.5, 
    restart_xentropy_max_mult = 3.0,
    x_penalty_max=100,
    x_penalty_min=0.1,
)

# %% 
# for key in vars(history):
#     print(key)
print(history.target.shape)
tokenizer.batch_decode(history.ids[-1])
# %%
pareto = build_pareto_frontier(tokenizer, history)
vars(pareto).keys()
pareto.text
# %%
pareto = build_pareto_frontier(tokenizer, history)
print(pareto)

ordering = np.argsort(pareto.xentropy)
plt.scatter(pareto.xentropy, pareto.target, c='k', label='Pareto frontier')
for i, k in enumerate(ordering):
    plt.text(pareto.xentropy[k] + 0.05, pareto.target[k] + 0.05, pareto.text[k], fontsize=8, rotation=-8, va='top', color='black', alpha=1.0)
# plt.xlim(3, 11)
# plt.ylim(0, 6)
plt.xlabel('Cross-entropy')
plt.ylabel('Activation')
plt.show()

# %%
