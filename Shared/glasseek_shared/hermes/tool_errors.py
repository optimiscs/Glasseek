"""Explicit failures from tool transports; webpage text is never reclassified."""
import json
import socket
import urllib.error


class ToolFailure(str):
    def __new__(cls, message, code='tool_execution_error', retryable=False, **details):
        payload=dict(success=False,error=str(message),error_code=code,retryable=retryable,**details)
        value=super().__new__(cls,json.dumps(payload,ensure_ascii=False))
        value.payload=payload
        return value


def exception_failure(exc):
    if isinstance(exc,urllib.error.HTTPError):
        return ToolFailure(f'Tool service returned HTTP {exc.code}.','tool_http_error',
            exc.code in (408,429) or exc.code>=500,http_status=exc.code)
    reason=getattr(exc,'reason',exc)
    if isinstance(reason,(TimeoutError,socket.timeout)):
        return ToolFailure('Tool request timed out.','tool_timeout',True)
    if isinstance(reason,socket.gaierror):
        return ToolFailure('Tool service DNS resolution failed.','tool_dns_error',True)
    if isinstance(exc,(OSError,urllib.error.URLError)):
        return ToolFailure('Tool service connection failed ('+type(reason).__name__+').','tool_connection_error',True)
    return ToolFailure('Tool execution failed ('+type(exc).__name__+').')


def http_json(opener,request,timeout):
    try:
        response=opener.open(request,timeout=timeout)
    except urllib.error.HTTPError as exc:
        try:
            with exc:payload=json.load(exc)
        except (OSError,ValueError):
            return exception_failure(exc)
        if isinstance(payload,dict) and payload.get('error'):
            return ToolFailure(payload['error'],payload.get('error_code','tool_http_error'),
                payload.get('retryable',exc.code in (408,429) or exc.code>=500),http_status=exc.code)
        return exception_failure(exc)
    except (OSError,ValueError) as exc:
        return exception_failure(exc)
    try:
        with response:payload=json.load(response)
    except (OSError,ValueError) as exc:
        return ToolFailure('Tool service returned invalid JSON ('+type(exc).__name__+').','tool_invalid_response')
    if not isinstance(payload,dict) or not payload:
        return ToolFailure('Tool service returned an empty or invalid response.','tool_invalid_response')
    if payload.get('success') is False or payload.get('error'):
        return ToolFailure(payload.get('error') or 'Tool service reported failure.',
            payload.get('error_code','tool_service_error'),payload.get('retryable',False))
    return payload
