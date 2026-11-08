"""Public Hermes import path; implementation owned by glasseek_shared."""
import sys
from glasseek_shared.hermes import turn_retry_state as _implementation
sys.modules[__name__] = _implementation
