"""Public Hermes import path; implementation owned by glasseek_shared."""
import sys
from glasseek_shared.hermes import tool_result_classification as _implementation
sys.modules[__name__] = _implementation
