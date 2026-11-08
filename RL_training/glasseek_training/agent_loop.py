"""verl 0.9.1 AgentLoop adapter. One output envelope per ORIGINAL rollout."""
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput

from .reward import score_answer
from .rollout import RolloutSettings, SearchRollout
from .web_tools import WebTools


class CMSearchAgentLoop(AgentLoopBase):
    async def run(self, sampling_params, **kwargs):
        cm = self.config.cm_gspo
        settings = RolloutSettings(**dict(cm.context))
        tools = WebTools(timeout=cm.tool_timeout, max_chars=cm.tool_max_chars)

        async def generate(prompt, budget):
            params = dict(sampling_params)
            params["max_tokens"] = budget
            output = await self.server_manager.generate(
                request_id=uuid4().hex, prompt_ids=prompt, sampling_params=params)
            if output.stop_reason == "aborted":
                raise RuntimeError("Rollout generation was aborted; refusing a partial training sample")
            if output.log_probs is None:
                raise ValueError("CM-GSPO needs behavior logprobs; enable rollout.calculate_log_probs")
            return output.token_ids, output.log_probs

        loop = SearchRollout(self.tokenizer, generate, tools, settings, self.apply_chat_template_kwargs)
        messages = list(kwargs["raw_prompt"])
        from glasseek_shared.context.config import mode as context_mode
        if context_mode() == 'clm':
            from .official_clm import rollout
            result = await rollout(self.tokenizer, generate, tools, settings, messages, self.apply_chat_template_kwargs)
        else:
            result = await loop.run(messages)
        reference = kwargs["reward_model"]["ground_truth"]
        if hasattr(reference, "tolist"):
            reference = reference.tolist()
        question = "\n".join(str(m["content"]) for m in messages if m["role"] == "user")
        reward = (await score_answer(question, result["final_answer"], reference, cm.reward_mode)
                  if result["stop_reason"] == "answer" else 0.0)
        # Standard agent-loop postprocessing expects a single sequence. The last
        # segment is only a transport envelope; CMTrainer expands cm_segments.
        last = result["segments"][-1]
        url_audit = ({'cm_url_mode': tools.url_mode, 'cm_url_registry': tools.url_registry.snapshot()}
                     if tools.url_mode != 'raw' else {})
        if url_audit:
            from glasseek_shared.hermes.url_prompt import prompt_module
            url_audit['cm_url_prompt'] = {k: v for k, v in prompt_module(tools.url_mode).items() if k != 'text'}
        completion_audit = ({'cm_completion_protocol': result['completion_protocol']}
                            if result.get('completion_protocol') else {})
        return AgentLoopOutput(
            prompt_ids=last["prompt_ids"], response_ids=last["response_ids"],
            response_mask=last["response_mask"], response_logprobs=last["log_probs"],
            reward_score=reward, num_turns=result["num_turns"], metrics={},
            extra_fields={"cm_segments": result["segments"], "cm_reward": reward,
                          "cm_stop_reason": result["stop_reason"], "cm_compactions": result["compactions"],
                          "cm_final_answer": result["final_answer"], "cm_official_clm_commit": result.get("official_clm_commit"), "cm_official_clm_artifacts": result.get("official_clm_artifacts"), "cm_context_edits": result.get("context_edits", []), **completion_audit, **url_audit})
