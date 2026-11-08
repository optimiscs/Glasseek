"""Producer-neutral request context accounting for Hermes SessionDB."""
import json,math

def breakdown(messages, tools):
    def count(text):
        # Classification is estimated text size, not additive API billing.
        # Template framing is deliberately excluded; server prompt_tokens wins.
        try:
            import tiktoken
            return len(tiktoken.get_encoding('cl100k_base').encode(text,disallowed_special=()))
        except (ImportError,ValueError,OSError):return math.ceil(len(text)/4)
    system=[m for m in messages if m.get('role') in ('system','developer')]
    other=[m for m in messages if m.get('role') not in ('system','developer')]
    text=lambda value:json.dumps(value,ensure_ascii=False,separators=(',',':'))
    result={'systemTokens':count(text(system)) if system else 0,
            'messageTokens':count(text(other)) if other else 0,
            'breakdownEstimated':True,'breakdownMethod':'serialized_request_text_excludes_template'}
    if tools is not None:result['toolTokens']=count(text(tools)) if tools else 0
    return result
