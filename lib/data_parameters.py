"""
Utilities for parameter extraction from data layout.

Parameters include the amount of keys, length of strings (both keys and values),
length of lists, ane more.
"""

def _string_value_length(json):
    if type(json) == str:
        return len(json)

    if type(json) == dict:
        return sum(map(_string_value_length, json.values()))

    if type(json) == list:
        return sum(map(_string_value_length, json))

    return 0

def _string_key_length(json):
    if type(json) == dict:
        return sum(map(len, json.keys())) + sum(map(_string_key_length, json.values()))

    return 0

def _num_keys(json):
    if type(json) == dict:
        return len(json.keys()) + sum(map(_num_keys, json.values()))

    return 0

def _num_objects(json):
    if type(json) == dict:
        return 1 + sum(map(_num_objects, json.values()))

    return 0

def json_to_param(json):
    """Return numeric parameters describing the structure of JSON data."""

    ret = dict()

    ret['strlen_keys'] = _string_key_length(json)
    ret['strlen_values'] = _string_value_length(json)
    ret['num_keys'] = _num_keys(json)
    ret['num_objects'] = _num_objects(json)

    return ret
