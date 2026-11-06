"""Fail early on options that would silently change the CM-GSPO objective."""
from .rollout import RolloutSettings


def require_verl_version():
    import verl
    if verl.__version__ != "0.9.1":
        raise RuntimeError(f"This recipe targets verl 0.9.1, found {verl.__version__}")


def validate_cm_config(config):
    a = config.actor_rollout_ref.actor
    r = config.actor_rollout_ref.rollout
    c = config.cm_gspo
    RolloutSettings(**dict(c.context)).validate()
    required = [
        (a.strategy in ("fsdp", "fsdp2"), "This recipe supports FSDP/FSDP2"),
        (a.policy_loss.loss_mode == "cm_gspo", "Use the paper-exact CM-GSPO loss"),
        (config.actor_rollout_ref.model.external_lib == "glasseek_training.verl_loss",
         "Register CM-GSPO on every worker through model.external_lib"),
        (a.clip_ratio_low == a.clip_ratio_high == c.clip_epsilon,
         "Paper Eq.(6) requires one symmetric clip_epsilon"),
        (0 < c.clip_epsilon < 1, "clip_epsilon must be between zero and one"),
        (a.loss_scale_factor is None, "Extra loss scaling is not part of Eq.(6)"),
        (not config.actor_rollout_ref.model.mtp.enable_train, "MTP loss is not part of Eq.(6)"),
        (config.algorithm.norm_adv_by_std_in_grpo, "Eq.(18) normalizes advantages by group std"),
        (a.loss_agg_mode == "seq-mean-token-mean", "GSPO needs per-segment token means"),
        (a.ppo_epochs == 1, "Each rollout batch must perform exactly one optimizer step"),
        (a.ppo_mini_batch_size == config.data.train_batch_size, "Use the full prompt batch per update"),
        (not a.use_kl_loss and a.entropy_coeff == 0, "This recipe implements the unregularized paper objective"),
        (not config.algorithm.use_kl_in_reward, "KL reward shaping is unsupported"),
        (config.algorithm.adv_estimator == "grpo", "Use rollout-level GRPO advantages"),
        (not config.reward.reward_model.enable, "Rewards are computed in the CM agent loop"),
        (not config.distillation.enabled, "Distillation is unsupported"),
        (r.name == "vllm" and r.mode == "async", "Use async vLLM rollouts"),
        (r.n >= 2 and r.calculate_log_probs, "Need grouped rollouts and behavior logprobs"),
        (r.temperature == 1 and r.top_p == 1 and r.top_k == -1,
         "Use temperature=1, top_p=1, top_k=-1 so behavior/current likelihoods match"),
        (r.logprobs_mode == "raw_logprobs", "Use raw behavior logprobs"),
        (r.agent.default_agent_loop == "cm_search", "Use the compaction-aware agent loop"),
        (r.max_model_len == c.context.max_model_len, "Model context limits must agree"),
        (config.data.max_prompt_length == c.context.max_prompt_length, "Prompt limits must agree"),
        (config.data.max_response_length == c.context.max_response_length, "Response limits must agree"),
    ]
    for ok, message in required:
        if not ok:
            raise ValueError(message)


def validate_environment(config):
    import os
    mode = config.cm_gspo.reward_mode
    if mode not in ("judge", "exact_match"):
        raise ValueError("cm_gspo.reward_mode must be judge or exact_match")
    required = ["SERPER_API_KEY"]
    if mode == "judge":
        required += ["GLASSEEK_JUDGE_BASE_URL", "GLASSEEK_JUDGE_MODEL", "GLASSEEK_JUDGE_API_KEY"]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise ValueError("Missing rollout environment variables: " + ", ".join(missing))
