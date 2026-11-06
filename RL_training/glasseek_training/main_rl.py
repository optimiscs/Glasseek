"""Launch CM-GSPO using verl 0.9.1's supported custom TaskRunner hook."""
import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.main_ppo import run_ppo
from verl.trainer.main_ppo_v0 import BaseTaskRunner
from verl.trainer.ppo.utils import create_rl_dataset, create_rl_sampler
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.config import validate_config
from verl.utils.fs import copy_to_local
from verl.utils.dataset.rl_dataset import collate_fn

from .trainer import CMTrainer
from .validation import validate_cm_config, require_verl_version, validate_environment


class CMTaskRunner(BaseTaskRunner):
    def run(self, config):
        require_verl_version()
        validate_cm_config(config)
        validate_environment(config)
        OmegaConf.resolve(config)
        _, group_cls = self.add_actor_rollout_worker(config)
        validate_config(config=config, use_reference_policy=False, use_critic=False)
        local = copy_to_local(config.actor_rollout_ref.model.path)
        tokenizer = hf_tokenizer(local, trust_remote_code=config.data.trust_remote_code)
        processor = hf_processor(local, trust_remote_code=config.data.trust_remote_code, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        pool = self.init_resource_pool_mgr(config)
        train = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor,
                                  max_samples=config.data.train_max_samples)
        val = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor,
                                is_train=False, max_samples=config.data.val_max_samples)
        trainer = CMTrainer(config=config, tokenizer=tokenizer, processor=processor,
                            role_worker_mapping=self.role_worker_mapping, resource_pool_manager=pool,
                            ray_worker_group_cls=group_cls, train_dataset=train, val_dataset=val,
                            collate_fn=collate_fn, train_sampler=create_rl_sampler(config.data, train))
        trainer.init_workers()
        trainer.fit()


@hydra.main(config_path="../configs", config_name="cm_gspo", version_base=None)
def main(config):
    require_verl_version()
    validate_cm_config(config)
    validate_environment(config)
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(CMTaskRunner))


if __name__ == "__main__":
    main()
