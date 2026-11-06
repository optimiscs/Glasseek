"""Worker-side registration loaded by HFModelConfig.external_lib on every rank."""
from verl.trainer.ppo.core_algos import register_policy_loss
from .loss import cm_gspo_loss


@register_policy_loss("cm_gspo")
def compute_policy_loss_cm_gspo(old_log_prob, log_prob, advantages, response_mask,
                                loss_agg_mode="seq-mean-token-mean", config=None,
                                rollout_is_weights=None):
    if config is None or loss_agg_mode != "seq-mean-token-mean":
        raise ValueError("CM-GSPO requires actor config and per-segment token means")
    if rollout_is_weights is not None:
        raise ValueError("Extra importance weights are not part of paper Eq.(6)")
    low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio
    if low != high:
        raise ValueError("Paper Eq.(6) uses one symmetric clipping epsilon")
    info = config.global_batch_info
    # No fallback to local row count: it would bias unequal compaction counts.
    if not info.get("global_batch_size") or not info.get("dp_size"):
        raise ValueError("Missing global original-rollout denominator or DP size")
    return cm_gspo_loss(old_log_prob, log_prob, advantages, response_mask,
                        rollout_count=info["global_batch_size"], dp_size=info["dp_size"],
                        clip_epsilon=low)
