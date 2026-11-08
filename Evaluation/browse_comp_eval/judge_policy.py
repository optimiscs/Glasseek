"""Scheduling policies; scoring and fingerprint generation remain in judge.py."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RecordJudgePolicy:
    workers: int = 4
    retry_seconds: float = 60

    def terminal(self, status, raw):
        return status in ('success', 'failed', 'policy_violation') and raw.get('attempt_consumed') is not False

    def pending(self, identity, prior, busy, retry_after, now):
        if prior and prior.get('pred_fingerprint') == identity[1] and prior.get('status') in ('judged', 'inference_error'):
            return False
        return identity not in busy and now >= retry_after.get(identity, 0)


@dataclass(frozen=True)
class SummaryJudgePolicy:
    retry_seconds: float = 60

    def due(self, signature, previous, now, last_run):
        return bool(signature) and (signature != previous or now - last_run >= self.retry_seconds)
