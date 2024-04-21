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
# from tqdm import tqdm
# from sae_lens.training.activations_store import ActivationsStore


# def get_tokens(
#     activation_store: ActivationsStore,
#     n_batches_to_sample_from: int = 2**10,
#     n_prompts_to_select: int = 4096 * 6,
# ):
#     all_tokens_list = []
#     pbar = tqdm(range(n_batches_to_sample_from))
#     for _ in pbar:
#         batch_tokens = activation_store.get_batch_tokens()
#         batch_tokens = batch_tokens[torch.randperm(batch_tokens.shape[0])][
#             : batch_tokens.shape[0]
#         ]
#         all_tokens_list.append(batch_tokens)

#     all_tokens = torch.cat(all_tokens_list, dim=0)
#     all_tokens = all_tokens[torch.randperm(all_tokens.shape[0])]
#     return all_tokens[:n_prompts_to_select]


# all_tokens = get_tokens(activations_store)  # should take a few minutes
# get_tokens()


# %%
# owt_data = load_dataset("stas/openwebtext-10k")
# tokenized_data = utils.tokenize_and_concatenate(
#     owt_data["train"], model.tokenizer, max_length=128
# )
# # %%
# all_tokens = tokenized_data["tokens"].cuda()
# # %%
# import os
# import sae_vis

# # reload(sae_vis)


# sparse_autoencoder = saes[1]
# # sparse_autoencoder.state_dict = lambda x: {}

# # test_feature_idx_gpt = list(range(10))
# # test_feature_idx_gpt = df[df["exp"]==str(2**k)]["f_id"].to_list()[:top_k]
# test_feature_idx_gpt =
# feature_vis_config_gpt = sae_vis.data_config_classes.SaeVisConfig(
#     hook_point=sparse_autoencoder.cfg.hook_name,
#     features=test_feature_idx_gpt,
#     batch_size=2048,
#     minibatch_size_tokens=128,
#     verbose=True,
# )

# sae_vis_data_gpt = sae_vis.data_storing_fns.SaeVisData.create(
#     encoder=sparse_autoencoder,
#     model=model,
#     tokens=all_tokens,  # type: ignore
#     cfg=feature_vis_config_gpt,
# )
# FEATURE_ROOT = Path("/workspace/SAE-Feature-Splitting/dashboards")
# # os.makedirs(FEATURE_ROOT/f"feature_vis_d_sae_{2**k}", exist_ok=True)
# filename = FEATURE_ROOT/f"exp_{2**k}_dash.html"
# sae_vis_data_gpt.save_feature_centric_vis(filename)
# # %%
# batch_size = 1024
# residuals = []
# for i in tqdm(range(0, 20_000, batch_size)):
#     batch = all_tokens[i : i + batch_size]
#     residuals.append(get_resid(batch))
# residuals = torch.cat(residuals, dim=0)
# residuals.shape, residuals.numel()
# # %%
# residuals = einops.rearrange(residuals, "batch pos d_model -> (batch pos) d_model")
# # %%
# df.sort_values("abs_sim", ascending=False)
# # %%
# vec1 = saes[1].W_dec[1082]
# # vec2 = saes[1].W_dec[723]
# vec2 = probe
# px.scatter(
#     y=to_numpy(residuals[::17] @ vec1),
#     x=to_numpy(residuals[::17] @ vec2),
#     marginal_x="histogram",
#     marginal_y="histogram",
#     trendline="ols",
#     labels={"y": "Decoder", "x": "Probe"},
#     opacity=0.2
# )
# # %%
# nutils.cos(vec1, probe)
# # %%
# import sys
# sys.path.append("/workspace/eliciting-latent-sentiment/")
# from prompt_utils import get_dataset, get_logit_diff
# import transformer_lens
# # model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")

# (
#     all_prompts, answer_tokens, clean_tokens, corrupted_tokens
# ) = get_dataset(model, "cuda", n_pairs = 1)

# %%
positive_adjs = [
  "amazing",
  "awesome",
  "beautiful",
  "brilliant",
  "exceptional",
  "extraordinary",
  "fabulous",
  "fantastic",
  "good",
  "great",
  "incredible",
  "lovely",
  "marvelous",
  "outstanding",
  "perfect",
  "remarkable",
  "spectacular",
  "superb",
  "terrific",
  "wonderful",
]
positive_verbs = [
  "adored",
  "admired",
  "appreciated",
  "applauded",
  "approved",
  "cherished",
  "commended",
  "congratulated",
  "embraced",
  "endorsed",
  "enjoyed",
  "exalted",
  "favored",
  "glorified",
  "liked",
  "loved",
  "praised",
  "relished",
  "respected",
  "savored",
  "treasured",
  "valued",
  "welcomed",
]
negative_adjs = [
  "awful",
  "bad",
  "disappointing",
  "disgusting",
  "dreadful",
  "horrendous",
  "horrible",
  "mediocre",
  "miserable",
  "offensive",
  "terrible",
  "unpleasant",
  "wretched",
]
negative_verbs = [
  "abhorred",
  "abominated",
  "condemned",
  "criticized",
  "denounced",
  "detested",
  "disdained",
  "disliked",
  "disparaged",
  "disrespected",
  "execrated",
  "hated",
  "loathed",
  "reviled",
  "scorned",
  "shunned",
  "spurned",
  "vilified",
]

# %%
TEMPLATE = "I thought this movie was {0}, I {1} it.\nConclusion: This movie is"

n = 100

pos_prompts = []
neg_prompts = []
for i in range(n):
    pos_prompts.append(TEMPLATE.format(random.choice(positive_adjs), random.choice(positive_verbs)))
    neg_prompts.append(TEMPLATE.format(random.choice(negative_adjs), random.choice(negative_verbs)))
pos_tokens = model.to_tokens(pos_prompts)
neg_tokens = model.to_tokens(neg_prompts)
ADJ_INDEX = 6
pos_verb_index = (pos_tokens==13).nonzero()[:, 1]-2
neg_verb_index = (neg_tokens==13).nonzero()[:, 1]-2

# %%
pos_logits, pos_cache = model.run_with_cache(pos_tokens, names_filter="blocks.7.hook_resid_post")
neg_logits, neg_cache = model.run_with_cache(neg_tokens, names_filter="blocks.7.hook_resid_post")
# %%
pos_resids = pos_cache["resid_post", 7]
pos_adj_resids = pos_resids[:, ADJ_INDEX]
pos_verb_resids = pos_resids[np.arange(100), pos_verb_index]
neg_resids = neg_cache["resid_post", 7]
neg_adj_resids = neg_resids[:, ADJ_INDEX]
neg_verb_resids = neg_resids[np.arange(100), neg_verb_index]
# %%
# line([(pos_logits[np.arange(100), (pos_tokens!=model.tokenizer.eos_token_id).sum(-1)].log_softmax(dim=-1)[:, model.to_single_token(" great")]), (neg_logits[np.arange(100), (neg_tokens!=model.tokenizer.eos_token_id).sum(-1)].log_softmax(dim=-1)[:, model.to_single_token(" great")])])
# # %%
# movie_tokens = model.to_tokens(all_prompts)
# print(model.to_str_tokens(movie_tokens[0]))
# print(model.to_str_tokens(movie_tokens[1]))
# movie_resids = get_resid(movie_tokens)
# movie_resids.shape
# # %%
# adj_resids = movie_resids[:, 7, :]
# line(adj_resids @ probe)
# # %%
# df.exp = df.exp.astype(int)
# df["sae"] = np.log2(df.exp).astype(int)
# # df["abs_sim"] = df.sim.abs()
# df_sorted = df.sort_values("sim", ascending=False, key=lambda n: n.abs())
# df_sorted
# # %%
# from sklearn.metrics import roc_curve
# sae_id = 1
# f_id = 723
# sae = saes[sae_id]
# v = sae.W_enc[:, f_id]
# v = v/v.norm()
# print(nutils.cos(v, probe))
# proj_adj_resids = to_numpy((adj_resids - sae.b_dec) @ v + sae.b_enc[f_id])
# histogram(np.stack([proj_adj_resids[::2], proj_adj_resids[1::2]], axis=1), barmode="overlay", marginal="box")
# fpr, tpr, thresholds = roc_curve([1, -1]*(len(movie_resids)//2), proj_adj_resids)
# line(x=fpr, y=tpr)
# # %%
# _, cache = sae.run_with_cache(adj_resids)
# cache["hook_sae_acts_pre"]

# # %%
# cache["hook_sae_acts_pre"][:, f_id]

# # %%
# line(cache["hook_sae_acts_post"][::2].mean(0) - cache["hook_sae_acts_post"][1::2].mean(0))

# # %%
# f = 1506
# histogram(torch.stack([cache["hook_sae_acts_post"][::2, f], cache["hook_sae_acts_post"][1::2, f]], axis=1), nbins=100)

# # %%
# margin = torch.maximum(
#     cache["hook_sae_acts_post"][::2].min(dim=0).values
#     - cache["hook_sae_acts_post"][1::2].max(dim=0).values,
#     cache["hook_sae_acts_post"][1::2].min(dim=0).values
#     - cache["hook_sae_acts_post"][::2].max(dim=0).values,
# )
# px.histogram(to_numpy(margin), hover_name=np.arange(2*768), marginal="box")
# %%
import sklearn.metrics
def tmr_adj_probing(vec):
    vec = vec/vec.norm()
    pos_proj = pos_adj_resids @ vec
    neg_proj = neg_adj_resids @ vec
    mean_diff = (pos_proj.mean() - neg_proj.mean()).abs().item()
    margin = max((pos_proj.min() - neg_proj.max()).item(), (neg_proj.min() - pos_proj.max()).item())
    roc_auc = sklearn.metrics.roc_auc_score([1]*n+[-1]*n, to_numpy(torch.cat([pos_proj, neg_proj])))
    return {"adj_mean_diff": mean_diff, "adj_margin": margin, "adj_roc_auc": roc_auc}
print(tmr_adj_probing(probe))

def tmr_verb_probing(vec):
    vec = vec/vec.norm()
    pos_proj = pos_verb_resids @ vec
    neg_proj = neg_verb_resids @ vec
    mean_diff = (pos_proj.mean() - neg_proj.mean()).abs().item()
    margin = max((pos_proj.min() - neg_proj.max()).item(), (neg_proj.min() - pos_proj.max()).item())
    roc_auc = sklearn.metrics.roc_auc_score([1]*n+[-1]*n, to_numpy(torch.cat([pos_proj, neg_proj])))
    return {"verb_mean_diff": mean_diff, "verb_margin": margin, "verb_roc_auc": roc_auc}
print(tmr_verb_probing(probe))
# %%
def directional_patching_hook(resid_post, hook, vec):
    assert len(vec.shape)==1
    pos_resid, neg_resid = einops.rearrange(resid_post, "(x batch) pos d_model -> x batch pos d_model", x=2)
    pos_adj_resid = pos_resid[:, ADJ_INDEX]
    pos_adj_resid_proj = (pos_adj_resid @ vec)[..., None] * vec
    neg_adj_resid = neg_resid[:, ADJ_INDEX]
    neg_adj_resid_proj = (neg_adj_resid @ vec)[..., None] * vec
    pos_adj_resid_new = (pos_adj_resid - pos_adj_resid_proj) + neg_adj_resid_proj
    neg_adj_resid_new = (neg_adj_resid - neg_adj_resid_proj) + pos_adj_resid_proj

    pos_verb_resid = pos_resid[np.arange(n), pos_verb_index]
    pos_verb_resid_proj = (pos_verb_resid @ vec)[..., None] * vec
    neg_verb_resid = neg_resid[np.arange(n), neg_verb_index]
    neg_verb_resid_proj = (neg_verb_resid @ vec)[..., None] * vec
    pos_verb_resid_new = (pos_verb_resid - pos_verb_resid_proj) + neg_verb_resid_proj
    neg_verb_resid_new = (neg_verb_resid - neg_verb_resid_proj) + pos_verb_resid_proj

    pos_resid[:, ADJ_INDEX] = pos_adj_resid_new
    pos_resid[np.arange(n), pos_verb_index] = pos_verb_resid_new

    neg_resid[:, ADJ_INDEX] = neg_adj_resid_new
    neg_resid[np.arange(n), neg_verb_index] = neg_verb_resid_new

HOOK_NAME = utils.get_act_name("resid_post", 7)

def tmr_patching(vec, do_patch=True, get_logit_diffs = False):
    all_tokens = torch.zeros((2*n, max(pos_tokens.shape[1], neg_tokens.shape[1])), dtype=int).cuda()
    all_tokens[:] = model.tokenizer.eos_token_id
    all_tokens[:n, :pos_tokens.shape[1]] = pos_tokens
    all_tokens[n:, :neg_tokens.shape[1]] = neg_tokens
    if do_patch:
        all_logits = model.run_with_hooks(all_tokens, fwd_hooks=[(HOOK_NAME, partial(directional_patching_hook, vec=vec))])
    else:
        all_logits = model(all_tokens)
    final_token_index = (all_tokens != model.tokenizer.eos_token_id).sum(-1)
    final_logits = all_logits[np.arange(2*n), final_token_index]
    final_log_prob = final_logits.log_softmax(dim=-1)
    pos_logit_diff = (final_log_prob[:n, model.to_single_token(" terrible")] - final_log_prob[:n, model.to_single_token(" great")])
    neg_logit_diff = (final_log_prob[n:, model.to_single_token(" great")] - final_log_prob[n:, model.to_single_token(" terrible")])
    # line([pos_logit_diff, neg_logit_diff])
    line([final_log_prob[:, model.to_single_token(" great")], final_log_prob[:, model.to_single_token(" terrible")]], line_labels=["great", "terrible"])
    if get_logit_diffs:
        return pos_logit_diff, neg_logit_diff
    else:
        acc = ((pos_logit_diff > 0).float().mean() + (neg_logit_diff > 0).float().mean()).item()/2
        logit_diff = ((pos_logit_diff).mean() + (neg_logit_diff).mean()).item()/2
        pos_normalised_logit_diff = (pos_logit_diff - base_pos_logit_diff) / (-base_neg_logit_diff - base_pos_logit_diff)
        neg_normalised_logit_diff = (neg_logit_diff - base_neg_logit_diff) / (-base_pos_logit_diff - base_neg_logit_diff)
        line([pos_normalised_logit_diff, neg_normalised_logit_diff], line_labels=["pos", "neg"])
        norm_logit_diff = (pos_normalised_logit_diff.mean() + neg_normalised_logit_diff.mean()).item()/2
        return {
            "patch_acc": acc,
            "patch_logit_diff": logit_diff,
            "patch_norm_logit_diff": norm_logit_diff,
        }
base_pos_logit_diff, base_neg_logit_diff = tmr_patching(probe, False, True)
print(tmr_patching(probe))
# %%
negation_prompts = [
    "You never fail",
    "You always fail",
    "Don't doubt",
    "Do doubt",
    "I really like",
    "I don't like",
]
negation_tokens = model.to_tokens(negation_prompts, padding_side="left")
negation_resids = get_resid(negation_tokens)
negation_resids = negation_resids[:, -1]
def negation_probe(vec):
    vec = vec/vec.norm()
    proj = negation_resids @ vec
    proj = proj[::2] - proj[1::2]
    return {"negation_mean": proj.mean().item(), "negation_acc": (proj>0).float().mean().item()}
print(negation_probe(probe))
# %%
def act_add(vec, coeff=10.0):
    vec = vec/vec.norm()
    def act_add_hook(act, hook):
        if act.shape[1]>1:
            act[:, 1:-1, :] -= coeff*vec
        return act
    prompt = "I really enjoyed the movie, in fact I loved it. I thought the movie was just very"
    model.blocks[7].hook_resid_post.add_hook(act_add_hook)
    s = model.generate(prompt, max_new_tokens=10, verbose=False)
    model.reset_hooks()
    return s
for i in range(10):
    print(act_add(probe, 10))
# %%
vec = probe
coeff = 100
vec = vec/vec.norm()
def act_add_hook(act, hook):
    if act.shape[1]>1:
        act[:, 1:-1, :] -= coeff*vec
    return act
model.blocks[7].hook_resid_post.add_hook(act_add_hook)
utils.test_prompt("I really enjoyed the movie, in fact I loved it. I thought the movie was just very", "bad", model)
model.reset_hooks()
utils.test_prompt("I really enjoyed the movie, in fact I loved it. I thought the movie was just very", "bad", model)
# %%
def get_all_metrics(vec):
    return {**tmr_adj_probing(vec), **tmr_verb_probing(vec), **tmr_patching(vec), **negation_probe(vec)}


# %%
cols = {}
cols["probe"] = get_all_metrics(probe)
cols["S1F723D"] = get_all_metrics(saes[1].W_dec[723])
cols["S1F723E"] = get_all_metrics(saes[1].W_enc[:, 723])
cols["rand1"] = get_all_metrics(torch.randn_like(probe))
cols["rand2"] = get_all_metrics(torch.randn_like(probe))
cols["rand3"] = get_all_metrics(torch.randn_like(probe))
nutils.show_df(pd.DataFrame(cols).T)
# %%
def directional_patching_hook_temp(resid_post, hook, vec):
    assert len(vec.shape) == 1
    pos_resid, neg_resid = einops.rearrange(
        resid_post, "(x batch) pos d_model -> x batch pos d_model", x=2
    )
    pos_adj_resid = pos_resid[:, ADJ_INDEX]
    pos_adj_resid_proj = (pos_adj_resid @ vec)[..., None] * vec
    neg_adj_resid = neg_resid[:, ADJ_INDEX]
    neg_adj_resid_proj = (neg_adj_resid @ vec)[..., None] * vec
    pos_adj_resid_new = (pos_adj_resid - pos_adj_resid_proj) + neg_adj_resid_proj
    neg_adj_resid_new = (neg_adj_resid - neg_adj_resid_proj) + pos_adj_resid_proj

    pos_verb_resid = pos_resid[np.arange(n), pos_verb_index]
    pos_verb_resid_proj = (pos_verb_resid @ vec)[..., None] * vec
    neg_verb_resid = neg_resid[np.arange(n), neg_verb_index]
    neg_verb_resid_proj = (neg_verb_resid @ vec)[..., None] * vec
    pos_verb_resid_new = (pos_verb_resid - pos_verb_resid_proj) + neg_verb_resid_proj
    neg_verb_resid_new = (neg_verb_resid - neg_verb_resid_proj) + pos_verb_resid_proj

    pos_resid[:, ADJ_INDEX] = pos_adj_resid_new
    pos_resid[np.arange(n), pos_verb_index] = pos_verb_resid_new

    neg_resid[:, ADJ_INDEX] = neg_adj_resid_new
    neg_resid[np.arange(n), neg_verb_index] = neg_verb_resid_new


HOOK_NAME = utils.get_act_name("resid_post", 7)


def tmr_patching_temp(vec, do_patch=True, get_logit_diffs=False):
    all_tokens = torch.zeros(
        (2 * n, max(pos_tokens.shape[1], neg_tokens.shape[1])), dtype=int
    ).cuda()
    all_tokens[:] = model.tokenizer.eos_token_id
    all_tokens[:n, : neg_tokens.shape[1]] = neg_tokens
    all_tokens[n:, : neg_tokens.shape[1]] = neg_tokens[torch.randperm(n)]
    if do_patch:
        all_logits = model.run_with_hooks(
            all_tokens,
            fwd_hooks=[(HOOK_NAME, partial(directional_patching_hook_temp, vec=vec))],
        )
    else:
        all_logits = model(all_tokens)
    final_token_index = (all_tokens != model.tokenizer.eos_token_id).sum(-1)
    final_logits = all_logits[np.arange(2 * n), final_token_index]
    final_log_prob = final_logits.log_softmax(dim=-1)
    pos_logit_diff = (
        final_log_prob[:n, model.to_single_token(" terrible")]
        - final_log_prob[:n, model.to_single_token(" great")]
    )
    neg_logit_diff = (
        final_log_prob[n:, model.to_single_token(" great")]
        - final_log_prob[n:, model.to_single_token(" terrible")]
    )
    # line([pos_logit_diff, neg_logit_diff])
    # line(
    #     [
    #         final_log_prob[:, model.to_single_token(" great")],
    #         final_log_prob[:, model.to_single_token(" terrible")],
    #     ],
    #     line_labels=["great", "terrible"],
    # )
    if get_logit_diffs:
        return pos_logit_diff, neg_logit_diff
    else:
        acc = (
            (pos_logit_diff > 0).float().mean() + (neg_logit_diff > 0).float().mean()
        ).item() / 2
        logit_diff = ((pos_logit_diff).mean() + (neg_logit_diff).mean()).item() / 2
        pos_normalised_logit_diff = (pos_logit_diff - base_pos_logit_diff) / (
            -base_neg_logit_diff - base_pos_logit_diff
        )
        neg_normalised_logit_diff = (neg_logit_diff - base_neg_logit_diff) / (
            -base_pos_logit_diff - base_neg_logit_diff
        )
        # line(
        #     [pos_normalised_logit_diff, neg_normalised_logit_diff],
        #     line_labels=["neg", "neg_shuff"],
        # )
        norm_logit_diff = (
            pos_normalised_logit_diff.mean() + neg_normalised_logit_diff.mean()
        ).item() / 2
        return {
            # "patch_acc": acc,
            # "patch_logit_diff": logit_diff,
            "patch_norm_logit_diff_pos": pos_normalised_logit_diff.mean().item(),
            "patch_norm_logit_diff_neg": neg_normalised_logit_diff.mean().item(),
        }

print(tmr_patching_temp(torch.randn_like(probe)))
print(tmr_patching_temp(probe))
print(tmr_patching_temp(torch.randn_like(probe)))

# %%
