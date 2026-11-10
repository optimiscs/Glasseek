#!/root/miniconda3/bin/python3
"""Publish a continuous token total from the original per-step TensorBoard events.

Runs independently of training and never edits the trainer's events. For repeated
steps after a rollback, the latest event is used; a changed already-published step
is treated as an error rather than silently publishing an incorrect total.
"""
import time
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

ROOT = Path('/root/autodl-tmp/Glasseek/reports/experiments/sft128k/tensorboard')
SOURCE = ROOT / 'length_curriculum'
DEST = ROOT / 'length_curriculum_total_tokens_corrected'
TAG = 'train/total_tokens(B)'


def read_tokens():
    accumulator = EventAccumulator(str(SOURCE), size_guidance={'scalars': 0}, purge_orphaned_data=False)
    accumulator.Reload()
    if 'train/global_tokens' not in accumulator.Tags()['scalars']:
        return {}
    latest = {}
    for item in accumulator.Scalars('train/global_tokens'):
        if item.step not in latest or item.wall_time >= latest[item.step].wall_time:
            latest[item.step] = item
    return latest


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    writer = EventFileWriter(str(DEST))
    published = {}
    total = 0
    print('Publishing corrected cumulative token totals', flush=True)
    try:
        while True:
            events = read_tokens()
            for step, count in published.items():
                if step in events and round(events[step].value) != count:
                    raise RuntimeError(f'Already published step {step} changed after rollback; manual rebuild required')
            previous_count = len(published)
            step = previous_count + 1
            while step in events:
                event = events[step]
                count = round(event.value)
                if count < 0:
                    raise ValueError(f'Invalid token count for step {step}: {count}')
                total += count
                published[step] = count
                writer.add_event(Event(wall_time=event.wall_time, step=step,
                                       summary=Summary(value=[Summary.Value(tag=TAG, simple_value=total / 1e9)])))
                step += 1
            writer.flush()
            if len(published) != previous_count:
                print(f'Published through step {len(published)}: {total} tokens', flush=True)
            time.sleep(15)
    finally:
        writer.close()


if __name__ == '__main__':
    main()
