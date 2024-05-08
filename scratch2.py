# %%
from neel.imports import *
from neel_plotly import *

torch.set_grad_enabled(False)
# %%
import wandb

run = wandb.init()
artifact = run.use_artifact(
    "jbloom/mats_sae_training_gpt2_feature_splitting_experiment/sparse_autoencoder_gpt2-small_blocks.8.hook_resid_pre_98304:v0",
    type="model",
)
artifact_dir = artifact.download()


# %%
def folder_to_file(folder):
    folder = Path(folder)
    files = list(folder.glob("*"))
    files = [str(f) for f in files]
    return files[0] if len(files) == 1 else files


from sae_lens.training.utils import BackwardsCompatiblePickleClass

file = folder_to_file(artifact_dir)
blob = torch.load(file, pickle_module=BackwardsCompatiblePickleClass)
config_dict = blob["cfg"].__dict__
state_dict = blob["state_dict"]
# %%
from transformer_lens import HookedSAE, HookedSAEConfig, HookedSAETransformer

cfg = HookedSAEConfig(
    d_sae=config_dict["d_sae"],
    d_in=config_dict["d_in"],
    hook_name=config_dict["hook_point"],
    use_error_term=True,
    dtype=torch.float32,
    seed=None,
    device="cuda",
)
print(cfg)
sae = HookedSAE(cfg)
sae.load_state_dict(state_dict)

# %%
saes = []
for n in range(8):
    artifact = run.use_artifact(
        f"jbloom/mats_sae_training_gpt2_feature_splitting_experiment/sparse_autoencoder_gpt2-small_blocks.8.hook_resid_pre_{2**n * 768}:v0",
        type="model",
    )
    artifact_dir = artifact.download()
    file = folder_to_file(artifact_dir)
    blob = torch.load(file, pickle_module=BackwardsCompatiblePickleClass)
    config_dict = blob["cfg"].__dict__
    state_dict = blob["state_dict"]
    cfg = HookedSAEConfig(
        d_sae=config_dict["d_sae"],
        d_in=config_dict["d_in"],
        hook_name=config_dict["hook_point"],
        use_error_term=True,
        dtype=torch.float32,
        seed=None,
        device="cuda",
    )
    print(cfg)
    sae = HookedSAE(cfg)
    sae.load_state_dict(state_dict)
    saes.append(sae)

# %%
model = HookedTransformer.from_pretrained("gpt2-small")
HOOK_NAME = "blocks.8.hook_resid_pre"
# %%
for c, sae in enumerate(saes):
    print(c, sae.b_enc.shape)
# %%
cosine_sims_list = []

for i in range(7):
    sae_small = saes[i]
    sae_big = saes[i + 1]
    d_sae_small = len(sae_small.b_enc)
    d_sae_big = len(sae_big.b_enc)
    # print(sae_big.W_dec.shape)

    # Shape is [d_sae_small, d_sae_big]
    cosine_sims = nutils.cos(sae_small.W_dec[:, None, :], sae_big.W_dec[None, :, :])
    cosine_sims_list.append(cosine_sims)

    histogram(
        cosine_sims.max(dim=0).values,
        title=f"Max Cosine Sim over {d_sae_small} for {d_sae_big}",
    )
    histogram(
        cosine_sims.max(dim=1).values,
        title=f"Max Cosine Sim over {d_sae_big} for {d_sae_small}",
    )

# %%
from sae_lens import SparseAutoencoder

layer = 8  # pick a layer you want.
sparse_autoencoder = SparseAutoencoder.from_pretrained(
    "gpt2-small-res-jb", f"blocks.{layer}.hook_resid_pre"
)
# %%
data = load_dataset("stas/openwebtext-10k")
# %%
all_tokens = utils.tokenize_and_concatenate(data["train"], model.tokenizer)["tokens"]
# %%
all_tokens.shape
# %%
batch_size = 16
resid_cache = []
def caching_hook(resid_pre, hook):
    resid_cache.append(resid_pre)
for i in tqdm.trange(0, 1000, batch_size):
    _ = model.run_with_hooks(all_tokens[i:i+batch_size], fwd_hooks=[("blocks.7.hook_resid_post", caching_hook)], stop_at_layer=8)
all_residuals = torch.cat(resid_cache, dim=0)
all_residuals.shape

# %%
# all_residuals = all_residuals[:, 1:, :].reshape(-1, 768)
def run_sae_enc(sae, apply_relu=False, start=None, stop=None):
    pre_acts = ((all_residuals - sae.b_dec) @ sae.W_enc[:, start:stop]) + sae.b_enc[start:stop]
    if apply_relu:
        return F.relu(pre_acts)
    else:
        return pre_acts

def get_feature_freq(sae, n=100):
    d_sae = len(sae.b_enc)
    freq_list = []
    interval = d_sae // n
    for start in tqdm.trange(0, d_sae, interval):
        acts = run_sae_enc(sae, True, start, start+interval)
        freqs = (acts>0).float().mean(0)
        freq_list.append(freqs)
    return torch.cat(freq_list)

# %%
EPS = 1e-7
histogram((get_feature_freq(saes[0])+EPS).log10())
# %%
sweep_5_freqs = get_feature_freq(saes[5])
histogram((sweep_5_freqs+EPS).log10(), histnorm="percent", title="Sweep 25K SAE Freq Hist")
original_freqs = get_feature_freq(sparse_autoencoder)
histogram((original_freqs+EPS).log10(), histnorm="percent", title="Original 25K SAE Freq Hist")
# %%
sae_small = saes[0]
sae_big = saes[1]
d_sae_small = len(sae_small.b_enc)
d_sae_big = len(sae_big.b_enc)
# print(sae_big.W_dec.shape)

# Shape is [d_sae_small, d_sae_big]
cosine_sims = nutils.cos(sae_small.W_dec[:, None, :], sae_big.W_dec[None, :, :])
# cosine_sims_list.append(cosine_sims)

histogram(
    cosine_sims.max(dim=0).values,
    title=f"Max Cosine Sim over {d_sae_small} for {d_sae_big}",
)
histogram(
    cosine_sims.max(dim=1).values,
    title=f"Max Cosine Sim over {d_sae_big} for {d_sae_small}",
)
# %%
# batch_size = 16
# resid_cache = []
# def caching_hook(resid_pre, hook):
#     resid_cache.append(resid_pre)
# for i in tqdm.trange(0, 1000, batch_size):
#     _ = model.run_with_hooks(all_tokens[i:i+batch_size], fwd_hooks=[("blocks.7.hook_resid_post", caching_hook)], stop_at_layer=8)
# all_residuals = torch.cat(resid_cache, dim=0)
# all_residuals.shape
token_df = nutils.make_token_df(all_tokens[:1008, 1:], len_prefix=8, len_suffix=3)
token_df
# %%
big_sae_max_cos, big_sae_max_cos_index = cosine_sims.max(dim=0)
big_sae_df = pd.DataFrame({"max_cos": to_numpy(big_sae_max_cos), "max_cos_index": to_numpy(big_sae_max_cos_index)})
big_sae_df.sort_values("max_cos", ascending=False).head()
# %%
if len(all_residuals.shape)>2:
    all_residuals = all_residuals[:, 1:, :].reshape(-1, 768)
# %%

small_index = 94
small_acts = F.relu((all_residuals - sae_small.b_dec) @ sae_small.W_enc[:, small_index] + sae_small.b_enc[small_index])
big_index = 415
big_acts = F.relu((all_residuals - sae_big.b_dec) @ sae_big.W_enc[:, big_index] + sae_big.b_enc[big_index])
temp_df = copy.deepcopy(token_df)
temp_df["small_value"] = to_numpy(small_acts)
temp_df["big_value"] = to_numpy(big_acts)
print("Cos", nutils.cos(sae_small.W_dec[small_index], sae_big.W_dec[big_index]))
print("Small freq", (small_acts>0).float().mean())
print("big freq", (big_acts>0).float().mean())
temp_df = temp_df.sort_values("small_value", ascending=False)
nutils.show_df(temp_df.head(25))
temp_df = temp_df.sort_values("big_value", ascending=False)
nutils.show_df(temp_df.head(25))
px.scatter(temp_df, x="small_value", y="big_value", trendline="ols")
# %%
