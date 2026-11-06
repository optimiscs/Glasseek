"""CM-GSPO dataflow on verl's Ray workers, engine, optimizer and checkpoints."""
import uuid
import numpy as np
from omegaconf import OmegaConf

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils import tensordict_utils as tu
from verl.workers.utils.padding import left_right_2_no_padding
from verl.utils.tracking import Tracking

from .cm_gspo import Rollout, Segment, pack_rollouts


class CMTrainer(RayPPOTrainer):
    def _generate_cm(self, batch_dict, *, validate=False):
        batch = DataProto.from_single_dict(batch_dict)
        # A repeated question in a batch is a separate group, just as in verl GRPO.
        ids = [uuid.uuid4().hex for _ in range(len(batch))]
        batch.non_tensor_batch["uid"] = np.array(ids, dtype=object)
        generated = self._get_gen_batch(batch).repeat(
            repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        generated.meta_info.update(global_steps=self.global_steps, validate=validate,
                                   temperature=self.config.actor_rollout_ref.rollout.temperature)
        generated, padding = pad_dataproto_to_divisor(
            generated, self.config.actor_rollout_ref.rollout.agent.num_workers)
        output = unpad_dataproto(self.async_rollout_manager.generate_sequences(generated), padding)
        self.checkpoint_manager.sleep_replicas()
        n = self.config.actor_rollout_ref.rollout.n
        if len(output) != len(ids) * n:
            raise ValueError("Agent loop did not return exactly n rollouts per prompt")
        rollouts = [Rollout(ids[i // n], float(output.non_tensor_batch["cm_reward"][i]),
                            [Segment(**s) for s in output.non_tensor_batch["cm_segments"][i]])
                    for i in range(len(output))]
        return rollouts, output

    def _update_cm(self, rollouts):
        actor = self.config.actor_rollout_ref.actor
        tensors, info = pack_rollouts(
            rollouts, group_size=self.config.actor_rollout_ref.rollout.n,
            pad_token_id=self.tokenizer.pad_token_id,
            batch_multiple=self._get_dp_size(self.actor_rollout_wg, "actor"),
            max_model_len=self.config.cm_gspo.context.max_model_len)
        data = DataProto.from_dict(tensors=tensors)
        data.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        td = left_right_2_no_padding(data.to_tensordict())
        # One optimizer update across ALL segments. The two sizes deliberately
        # differ: dispatch/minibatch rows versus the Eq.(6) rollout denominator.
        tu.assign_non_tensor(td, global_batch_size=info["rollout_count"],
                             mini_batch_size=info["padded_rows"], epochs=1, seed=actor.data_loader_seed,
                             dataloader_kwargs={"shuffle": False}, compute_loss=True,
                             calculate_entropy=False, distillation_use_topk=False)
        output = self.actor_rollout_wg.update_actor(td)
        raw = tu.get(output, "metrics")
        metrics = {"actor/" + key: float(np.mean(value)) for key, value in raw.items()}
        metrics.update({"cm/" + key: value for key, value in info.items()})
        return metrics

    def _validate_cm(self):
        rewards = []
        for batch in self.val_dataloader:
            rollouts, _ = self._generate_cm(batch, validate=True)
            rewards.extend(r.reward for r in rollouts)
            self.checkpoint_manager.update_weights(self.global_steps)
        return {"val/reward": float(np.mean(rewards))}

    def fit(self):
        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))
        self.global_steps = 0
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)
        if self.config.trainer.val_before_train:
            logger.log(self._validate_cm(), step=self.global_steps)
        if self.config.trainer.val_only:
            return
        try:
            while self.global_steps < self.total_training_steps:
                for batch_dict in self.train_dataloader:
                    if self.global_steps >= self.total_training_steps:
                        break
                    rollouts, output = self._generate_cm(batch_dict)
                    metrics = self._update_cm(rollouts)
                    self.global_steps += 1
                    metrics["train/reward"] = float(np.mean([r.reward for r in rollouts]))
                    metrics["cm/compactions_mean"] = float(np.mean(output.non_tensor_batch["cm_compactions"]))
                    metrics["cm/completion_rate"] = float(np.mean(
                        output.non_tensor_batch["cm_stop_reason"] == "answer"))
                    final = self.global_steps == self.total_training_steps
                    save_freq = self.config.trainer.save_freq
                    if save_freq > 0 and (final or self.global_steps % save_freq == 0):
                        self._save_checkpoint()
                    self.checkpoint_manager.update_weights(self.global_steps)
                    test_freq = self.config.trainer.test_freq
                    if test_freq > 0 and (final or self.global_steps % test_freq == 0):
                        metrics.update(self._validate_cm())
                    logger.log(metrics, step=self.global_steps)
        finally:
            self._shutdown_dump_executor()
