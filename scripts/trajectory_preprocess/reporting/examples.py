"""Redaction helpers shared by traceable safety examples."""
import re
AUDIT_EMAIL=re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')


def safe_window(text,start,end,scanner):
    # Redact against the complete view BEFORE clipping: a clipped signed URL can
    # lose its scheme and evade the ordinary URL-auth detector.
    spans=sorted([(a,b) for a,b,_ in scanner.sensitive_spans(text)]+[(m.start(),m.end()) for m in AUDIT_EMAIL.finditer(text)]);merged=[]
    for a,b in spans:
        if merged and a<=merged[-1][1]:merged[-1][1]=max(b,merged[-1][1])
        else:merged.append([a,b])
    result=[];cursor=start
    for a,b in merged:
        a=max(a,start);b=min(b,end)
        if a>=b:continue
        result.extend([text[cursor:a],'[REDACTED]']);cursor=b
    result.append(text[cursor:end])
    return scanner.scrub(''.join(result))

