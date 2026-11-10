"""Shared search batch limits; providers still receive individual queries."""
import json
MAX_QUERIES = 5
VERSION = 'web-search-call-batch-v1'


def decode_query_array(value,max_layers=4):
    """Commit decoding only if the result is an array; quoted queries stay text."""
    original=value
    for _ in range(max_layers):
        if not isinstance(value,str) or not value.lstrip().startswith(('[','"')):break
        try:value=json.loads(value)
        except ValueError:break
        if isinstance(value,list):return value
    return original


def validate_queries(query):
    queries = [query] if isinstance(query, str) else query
    if (not isinstance(queries, list) or not 1 <= len(queries) <= MAX_QUERIES
        or any(not isinstance(q, str) or not q.strip() for q in queries)):
        raise ValueError('query must be a nonempty string or an array of 1 to 5 nonempty strings')
    return queries
