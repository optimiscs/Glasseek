"""Target-format rendering separate from the reversible source migration archive."""
import copy
import re
from runtime_metadata import clean_messages

REPORT_FINAL=re.compile(r'^(?P<prefix>(?:\s*<think>.*?</think>)*\s*)<report>\s*(?P<body>.*?)\s*</report>\s*$',re.S)

def target_messages(item):
    messages,events=clean_messages(item['messages']);transforms=[]
    if any(e['status']=='cleaned' for e in events):transforms.append('delegate_runtime_metadata_removed')
    if any(e['status']=='skipped' for e in events):transforms.append('delegate_runtime_metadata_parse_review')
    if item['agent_role']=='child' and messages and messages[-1]['role']=='assistant':
        text=messages[-1]['content'];match=REPORT_FINAL.fullmatch(text)
        if match and '<report>' not in match['body'] and '</report>' not in match['body']:
            # SearchSwarm strips this protocol envelope before returning a report.
            # Glasseek returns final_response directly after removing thinking.
            # Align the child's visible output with the report already in the parent.
            messages[-1]['content']=match['prefix']+match['body']
            transforms.append('child_final_report_envelope_removed')
    return messages,transforms
