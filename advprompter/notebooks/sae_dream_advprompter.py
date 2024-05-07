# flake8: noqa
# %%

import torch
import transformer_lens as tl 
import sae_lens as sl

torch.set_grad_enabled(False)

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"


# %%
# Basic loading
from sae_lens.training.session_loader import LMSparseAutoencoderSessionloader
from sae_lens.toolkit.pretrained_saes import get_gpt2_res_jb_saes

layer: int = 5
hook_point = f"blocks.{layer}.hook_resid_pre"

# if the SAEs were stored with precomputed feature sparsities,
#  those will be return in a dictionary as well.
saes, sparsities = get_gpt2_res_jb_saes(hook_point)
print(saes.keys())
sparse_autoencoder = saes[hook_point]
sparse_autoencoder.to(device)
sparse_autoencoder.cfg.device = device

loader = LMSparseAutoencoderSessionloader(sparse_autoencoder.cfg)

# don't overwrite the sparse autoencoder with the loader's sae (newly initialized)
(
    model, 
    _, activation_store
) = loader.load_sae_training_group_session()
model: tl.HookedTransformer = model

# TODO: We should have the session loader go straight to huggingface.
# %%
print(model)
print(sparse_autoencoder)
# %%

text = "I am a Jedi"

# Define a metric
def metric(
    text: str, 
    model: tl.HookedSAETransformer, 
    sae, 
    feature_idx: int = 23
):
    # Return layer 5, feature 23 activation
    _, cache = model.run_with_cache(text)
    sae_in = cache[sae.cfg.hook_point]
    # sae_out, feature_acts = sae(sae_in)[:2]
    hidden_pre = sae._encode_with_hidden_pre(sae_in)[1]
    return hidden_pre[..., feature_idx]
sae = sparse_autoencoder
metric(text, model, sae)
# %%


tl.utils.test_prompt(text, "Master", model)

k = 10
logits, _ = model.run_with_cache(text)
# Sample from the distribution
token_ids = torch.distributions.categorical.Categorical(logits=logits).sample()
print(token_ids)
model.to_str_tokens(token_ids)
# top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
# print(top_k_indices.shape)

# %%
from jaxtyping import Int, Float

def sample_from_logits(
    logits: Float[torch.Tensor, "n_beams seq_length vocab_size"],
    k: int = 16
) -> Int[torch.Tensor, "n_beams 1 k"]:
    """ Sample one token for every candidate """
    probs = torch.softmax(logits[:, -1], dim=-1)
    return torch.multinomial(probs, k).unsqueeze(1)

#  Sanity check
logits, _ = model.run_with_cache(text)
oup = sample_from_logits(logits)
print(oup.shape)

# %%
def sample_next_token(
    model: tl.HookedTransformer, 
    sae: sl.SparseAutoencoder,
    tokens: Int[torch.Tensor, "n_beams seq_length"], 
    k: int = 10,
    n_beams: int = 10
):
    # TODO: do we ever want sequences of different length
    logits, _ = model.run_with_cache(tokens)
    token_ids: Int[torch.Tensor, "n_beams 1 k"] = sample_from_logits(logits, k)
    candidates: Int[torch.Tensor, "n_beams seq_length"] = torch.cat((torch.repeat_interleave(tokens, k, 0), token_ids.view(-1, 1)), 1)
    metrics: Float[torch.Tensor, "n_beams seq_length"] = metric(candidates, model, sae)
    indices = metrics[:, -1].topk(min(n_beams, len(metrics))).indices
    return candidates[indices]

beams = model.to_tokens([""], prepend_bos=True)
print(beams.shape)
print(model.to_str_tokens(beams))
beams = sample_next_token(model, sae, beams)
print(beams.shape)
# NOTE: only works for one beam at a time
print(model.to_str_tokens(beams[0]))

# %%

text = "I am a Jedi"
metric(text, model, sae)

# %%
from tqdm.auto import trange

metrics_list = []
tokens_list = []
n_tokens_to_add: int = 10
query_text = "I am a"
beams = model.to_tokens([query_text], prepend_bos=True)
for _ in (bar := trange(n_tokens_to_add)):
    beams = sample_next_token(model, sae, beams)
    metrics = metric(beams, model, sae)
    metrics_list.append(metrics[0][-1].item())
    tokens_list.append(beams[0][-1].item())
    print(model.to_str_tokens(beams[0]))
    bar.set_postfix(metric=metrics[0][-1])

print(metrics_list)
print(tokens_list)

# %%
