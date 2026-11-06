"""Run and publish the 44 RL-4B BrowseComp evaluation."""
import sys
from judge_after_existing_run import ROOT
from supervise_evaluation import supervise

if __name__ == '__main__':
    sys.exit(supervise(
        run='glasseek_rl4_quest128_ssh44_c16', workspace_id='glasseek-rl4-browsecomp-128-c16',
        model='Glasseek RL-4B', label='Glasseek RL-4B · BrowseComp · Quest 128 · 4 并发',
        description='SSH 44 · RL-4B · TP1 · BF16 KV · DeepSeek Flash；单卡 4 Agent',
        script=ROOT / 'services/remote44/run_rl4_c16.sh', log_name='viewer-publisher-rl4-c16.log'))
