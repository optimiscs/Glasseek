"""Run and publish the local four-GPU RL-9B BrowseComp evaluation."""
import os
import sys
from pathlib import Path
from judge_after_existing_run import ROOT
from supervise_evaluation import supervise

if __name__ == '__main__':
    sys.exit(supervise(
        run='glasseek_rl9b_quest128_32', workspace_id='glasseek-rl9b-browsecomp-128-32',
        model='Glasseek RL-9B', label='Glasseek RL-9B · BrowseComp · Quest 128 · 32 并发',
        description='本机四卡 · RL-9B · TP1 · FP8 KV · DeepSeek Flash；单卡 8 Agent',
        script=Path(os.environ.get('GLASSEEK_RL9B_LAUNCHER', ROOT / 'services/run_rl9b_eval.sh')),
        log_name='viewer-publisher-rl9b-32.log'))
