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
probe = torch.tensor(
    np.load(
        "/workspace/SAE-Feature-Splitting/logistic_regression_simple_train_ADJ_layer7.npy"
    ),
    dtype=torch.float32,
).cuda()
probe = probe / probe.norm()


# probe_8 = torch.tensor(np.load("/workspace/SAE-Feature-Splitting/logistic_regression_simple_train_ADJ_layer8.npy"), dtype=torch.float32).cuda()
# probe_8 = probe_8 / probe_8.norm()
def get_resid(tokens):
    return model.run_with_cache(
        tokens, names_filter="blocks.7.hook_resid_post", stop_at_layer=8
    )[1]["blocks.7.hook_resid_post"][:, 1:]


# print(sample_resid)
# %%
# %%
sad_resid = get_resid("I feel really, really sad")
happy_resid = get_resid("I feel really, really happy")
line(torch.stack([sad_resid, happy_resid]) @ probe)
# %%
text = " presidential candidate Barack Obama"
line(get_resid(text) @ probe, x=model.to_str_tokens(text, prepend_bos=False))
# %%
cosine_sims = []
df = pd.DataFrame(columns=["sim", "exp", "f_id"])
for i, sae in enumerate(saes):
    cosine_sims.append(sae.W_dec @ probe)
    # line(cosine_sims[-1], title=f"Expansion {2**i}")
    x = to_numpy(cosine_sims[-1])
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                {"sim": x, "exp": [str(2**i)] * len(x), "f_id": range(len(x))}
            ),
        ]
    )
df
# %%
# px.histogram(df, histnorm="percent", barmode='overlay', x='sim', color='exp', marginal='box')
# %%
# px.violin(df, x='exp', y='sim', title="Cosine sim of SAE features with probe").update_xaxes(type="category")
# %%
from tqdm import tqdm
from sae_lens.training.activations_store import ActivationsStore


def get_tokens(
    activation_store: ActivationsStore,
    n_batches_to_sample_from: int = 2**10,
    n_prompts_to_select: int = 4096 * 6,
):
    all_tokens_list = []
    pbar = tqdm(range(n_batches_to_sample_from))
    for _ in pbar:
        batch_tokens = activation_store.get_batch_tokens()
        batch_tokens = batch_tokens[torch.randperm(batch_tokens.shape[0])][
            : batch_tokens.shape[0]
        ]
        all_tokens_list.append(batch_tokens)

    all_tokens = torch.cat(all_tokens_list, dim=0)
    all_tokens = all_tokens[torch.randperm(all_tokens.shape[0])]
    return all_tokens[:n_prompts_to_select]


# all_tokens = get_tokens(activations_store)  # should take a few minutes
# get_tokens()

# %%
owt_data = load_dataset("stas/openwebtext-10k")
tokenized_data = utils.tokenize_and_concatenate(
    owt_data["train"], model.tokenizer, max_length=128
)
# %%
all_tokens = tokenized_data["tokens"].cuda()
# %%
import os
import sae_vis

# reload(sae_vis)


# for k, v in enumerate(saes):
# print(k)
# print(v.cfg.hook_point)
# sparse_autoencoder = saes[k]
# sparse_autoencoder.state_dict = lambda x: {}
df["abs_sim"] = df["sim"].abs()
# df = df.sort_values("abs_sim", ascending=False)
# top_k = 5
# for k in range(8):
#     sparse_autoencoder = saes[k]
#     # test_feature_idx_gpt = list(range(10))
#     test_feature_idx_gpt = df[df["exp"]==str(2**k)]["f_id"].to_list()[:top_k]
#     feature_vis_config_gpt = sae_vis.data_config_classes.SaeVisConfig(
#         hook_point=sparse_autoencoder.cfg.hook_name,
#         features=test_feature_idx_gpt,
#         batch_size=2048,
#         minibatch_size_tokens=128,
#         verbose=True,
#     )

#     sae_vis_data_gpt = sae_vis.data_storing_fns.SaeVisData.create(
#         encoder=sparse_autoencoder,
#         model=model,
#         tokens=all_tokens,  # type: ignore
#         cfg=feature_vis_config_gpt,
#     )
#     FEATURE_ROOT = Path("/workspace/SAE-Feature-Splitting/dashboards")
#     # os.makedirs(FEATURE_ROOT/f"feature_vis_d_sae_{2**k}", exist_ok=True)
#     filename = FEATURE_ROOT/f"exp_{2**k}_dash.html"
#     sae_vis_data_gpt.save_feature_centric_vis(filename)
# %%
batch_size = 1024
residuals = []
for i in tqdm(range(0, 20_000, batch_size)):
    batch = all_tokens[i : i + batch_size]
    residuals.append(get_resid(batch))
residuals = torch.cat(residuals, dim=0)
residuals.shape, residuals.numel()
# %%
residuals = einops.rearrange(residuals, "batch pos d_model -> (batch pos) d_model")
# %%
df.sort_values("abs_sim", ascending=False)
# %%
vec1 = saes[1].W_dec[1082]
# vec2 = saes[1].W_dec[723]
vec2 = probe
px.scatter(
    y=to_numpy(residuals[::17] @ vec1),
    x=to_numpy(residuals[::17] @ vec2),
    marginal_x="histogram",
    marginal_y="histogram",
    trendline="ols",
    labels={"y": "Decoder", "x": "Probe"},
    opacity=0.2
)
# %%
nutils.cos(vec1, probe)
# %%
