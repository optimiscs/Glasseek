"""Explicit source selection shared by intake, publication and training."""
VERSION = 'dataset-selection-v1-no-openresearcher'
EXCLUDED_SOURCES = frozenset({'openresearcher', 'openresearcher-dataset'})


def excluded_source(source):
    return isinstance(source, str) and any(part.casefold() in EXCLUDED_SOURCES for part in source.split('/'))
