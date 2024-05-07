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


def neuron_runner(layer, neuron):
    def f(*model_args, **model_kwargs):
        out = {}

        def get_target(module, input, output):
            hidden_pre = sparse_autoencoder._encode_with_hidden_pre(
                input[0])[1]
            out["target"] = hidden_pre[..., neuron][:, -1:].mean(dim=-1)

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
# %%
# | output: false
runner = neuron_runner(layer=layer, neuron=234)
history = epo(
    runner, model, tokenizer, seed=1001,
    restart_xentropy = 0.5, restart_xentropy_max_mult = 3.0,
    x_penalty_max=100,
    x_penalty_min=0.1,
)

# %%
pareto = build_pareto_frontier(tokenizer, history)

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
