"""Strict wire schemas. IDs and coverage are owned by the program."""
VERSION = 'rules-first-curation-v1'


def obj(properties):
    return {'type':'object', 'properties':properties, 'required':list(properties), 'additionalProperties':False}


def enum(values):
    values = list(values)
    return {'type':'string', 'enum':values} if values else {'type':'string', 'enum':['__no_admissible_id__']}


def array(items, minimum=0, maximum=None):
    result = {'type':'array', 'items':items, 'minItems':minimum}
    if maximum is not None: result['maxItems'] = maximum
    return result


TEXT = {'type':'string', 'minLength':1}
BOOL = {'type':'boolean'}
STATUSES = enum(['supported','contradicted','insufficient','conflict'])
