import copy
import json
import re
from .edit import ContextState, propose_edit
from .config import config

REF = re.compile(r'<url>\s*((?:U[1-9]\d*|D[1-9]\d*@v[1-9]\d*(?:#E[1-9]\d*)?))\s*</url>')

class WorkingContext:
    def __init__(self):
        self.state = ContextState(0, '')
        self.events = []
        self.settings = config()

    def edit(self, args, observations, count_tokens, validate=None):
        visible = frozenset(ref for text in observations for ref in REF.findall(text))
        old = self.state
        result = propose_edit(old, base_version=args.get('base_version'), content=args.get('content'),
            visible_refs=visible, count_tokens=count_tokens, max_tokens=self.settings.max_note_tokens)
        reason = result.reason
        if result.accepted and validate is not None:
            try:
                if not validate(result.state.content):reason = 'request_over_budget'
            except (ValueError, TypeError):reason = 'invalid_request'
        accepted = result.accepted and reason == 'accepted'
        if accepted:self.state = result.state
        receipt = {'accepted':accepted,'reason':reason,'version':self.state.version,
                   'tokens_before':result.input_tokens,'tokens_after':result.proposed_tokens}
        self.events.append(copy.deepcopy(receipt))
        return json.dumps(receipt)

    def note(self):
        return {'role':'user','content':f'Working context v{self.state.version}:\n{self.state.content}\nContinue the original task.'}
