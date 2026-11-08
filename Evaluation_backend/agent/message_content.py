"""Public Hermes import path; implementation owned by glasseek_shared."""
import sys
from glasseek_shared.hermes import message_content as _implementation
sys.modules[__name__] = _implementation
