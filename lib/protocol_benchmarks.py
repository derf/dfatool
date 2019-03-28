import bson
import cbor
import json
import msgpack
import ubjson

import os
import re
import time
from filelock import FileLock

class DummyProtocol:
    def __init__(self):
        self.max_serialized_bytes = None
        self.enc_buf = ''
        self.dec_buf = ''
        self.dec_buf0 = ''
        self.dec_buf1 = ''
        self.dec_buf2 = ''
        self.dec_index = 0
        self.transition_map = dict()

    def assign_and_kout(self, signature, assignment, transition_args = None):
        self.new_var(signature)
        self.assign_var(assignment, transition_args = transition_args)
        self.kout_var()

    def new_var(self, signature):
        self.dec_index += 1
        self.dec_buf0 += '{} dec_{:d};\n'.format(signature, self.dec_index)

    def assign_var(self, assignment, transition_args = None):
        snippet = 'dec_{:d} = {};\n'.format(self.dec_index, assignment)
        self.dec_buf1 += snippet
        if transition_args:
            self.add_transition(snippet, transition_args)

    def get_var(self):
        return 'dec_{:d}'.format(self.dec_index)

    def kout_var(self):
        self.dec_buf2 += 'kout << dec_{:d};\n'.format(self.dec_index)

    def note_unsupported(self, value):
        note = '// value {} has unsupported type {}\n'.format(value, type(value))
        self.enc_buf += note
        self.dec_buf += note
        self.dec_buf1 += note

    def is_ascii(self):
        return True

    def get_encode(self):
        return ''

    def get_buffer_declaration(self):
        return ''

    def get_serialize(self):
        return ''

    def get_deserialize(self):
        return ''

    def get_decode_and_output(self):
        return ''

    def get_decode_vars(self):
        return ''

    def get_decode(self):
        return ''

    def get_decode_output(self):
        return ''

    def get_extra_files(self):
        return dict()

    def can_get_serialized_length(self):
        return False

    def add_transition(self, code_snippet: str, args: list):
        self.transition_map[code_snippet] = args
        return code_snippet

class ArduinoJSON(DummyProtocol):

    def __init__(self, data, bufsize = 255, int_type = 'uint16_t', float_type = 'float'):
        super().__init__()
        self.data = data
        self.max_serialized_bytes = self.get_serialized_length() + 2
        self.children = set()
        self.bufsize = bufsize
        self.int_type = int_type
        self.float_type = float_type
        self.enc_buf += self.add_transition('ArduinoJson::StaticJsonBuffer<{:d}> jsonBuffer;\n'.format(bufsize), [bufsize])
        self.enc_buf += 'ArduinoJson::JsonObject& root = jsonBuffer.createObject();\n'
        self.from_json(data, 'root')

    def get_serialized_length(self):
        return len(json.dumps(self.data))

    def can_get_serialized_length(self):
        return True

    def get_encode(self):
        return self.enc_buf

    def get_buffer_declaration(self):
        return 'char buf[{:d}];\n'.format(self.max_serialized_bytes)

    def get_buffer_name(self):
        return 'buf'

    def get_length_var(self):
        return 'serialized_size'

    def get_serialize(self):
        return self.add_transition('uint16_t serialized_size = root.printTo(buf);\n', [self.max_serialized_bytes])

    def get_deserialize(self):
        ret = self.add_transition('ArduinoJson::StaticJsonBuffer<{:d}> jsonBuffer;\n'.format(self.bufsize), [self.bufsize])
        ret += self.add_transition('ArduinoJson::JsonObject& root = jsonBuffer.parseObject(buf);\n', [self.max_serialized_bytes])
        return ret

    def get_decode_and_output(self):
        return 'kout << dec << "dec:";\n' + self.dec_buf + 'kout << endl;\n';

    def get_decode_vars(self):
        return self.dec_buf0

    def get_decode(self):
        return self.dec_buf1

    def get_decode_output(self):
        return 'kout << dec << "dec:";\n' + self.dec_buf2 + 'kout << endl;\n';

    def add_to_list(self, enc_node, dec_node, offset, value):
        if type(value) == str:
            if len(value) and value[0] == '$':
                self.enc_buf += '{}.add({});\n'.format(enc_node, value[1:])
                self.dec_buf += 'kout << {}[{:d}].as<{}>();\n'.format(dec_node, offset, self.int_type)
                self.assign_and_kout(self.int_type, '{}[{:d}].as<{}>()'.format(dec_node, offset, self.int_type))
            else:
                self.enc_buf += self.add_transition('{}.add("{}");\n'.format(enc_node, value), [len(value)])
                self.dec_buf += 'kout << {}[{:d}].as<const char *>();\n'.format(dec_node, offset)
                self.assign_and_kout('char const*', '{}[{:d}].as<char const *>()'.format(dec_node, offset), transition_args = [len(value)])

        elif type(value) == list:
            child = enc_node + 'l'
            while child in self.children:
                child += '_'
            self.enc_buf += 'ArduinoJson::JsonArray& {} = {}.createNestedArray();\n'.format(
                child, enc_node)
            self.children.add(child)
            self.from_json(value, child)

        elif type(value) == dict:
            child = enc_node + 'o'
            while child in self.children:
                child += '_'
            self.enc_buf += 'ArduinoJson::JsonObject& {} = {}.createNestedObject();\n'.format(
                child, enc_node)
            self.children.add(child)
            self.from_json(value, child)

        elif type(value) == float:
            self.enc_buf += '{}.add({});\n'.format(enc_node, value)
            self.dec_buf += 'kout << {}[{:d}].as<{}>();\n'.format(dec_node, offset, self.float_type)
            self.assign_and_kout(self.float_type, '{}[{:d}].as<{}>()'.format(dec_node, offset, self.float_type))

        elif type(value) == int:
            self.enc_buf += '{}.add({});\n'.format(enc_node, value)
            self.dec_buf += 'kout << {}[{:d}].as<{}>();\n'.format(dec_node, offset, self.int_type)
            self.assign_and_kout(self.int_type, '{}[{:d}].as<{}>()'.format(dec_node, offset, self.int_type))

        else:
            self.note_unsupported(value)

    def add_to_dict(self, enc_node, dec_node, key, value):
        if type(value) == str:
            if len(value) and value[0] == '$':
                self.enc_buf += self.add_transition('{}["{}"] = {};\n'.format(enc_node, key, value[1:]), [len(key)])
                self.dec_buf += 'kout << {}["{}"].as<{}>();\n'.format(dec_node, key, self.int_type)
                self.assign_and_kout(self.int_type, '{}["{}"].as<{}>()'.format(dec_node, key, self.int_type))
            else:
                self.enc_buf += self.add_transition('{}["{}"] = "{}";\n'.format(enc_node, key, value), [len(key), len(value)])
                self.dec_buf += 'kout << {}["{}"].as<const char *>();\n'.format(dec_node, key)
                self.assign_and_kout('char const*', '{}["{}"].as<const char *>()'.format(dec_node, key), transition_args = [len(key), len(value)])

        elif type(value) == list:
            child = enc_node + 'l'
            while child in self.children:
                child += '_'
            self.enc_buf += self.add_transition('ArduinoJson::JsonArray& {} = {}.createNestedArray("{}");\n'.format(
                child, enc_node, key), [len(key)])
            self.children.add(child)
            self.from_json(value, child, '{}["{}"]'.format(dec_node, key))

        elif type(value) == dict:
            child = enc_node + 'o'
            while child in self.children:
                child += '_'
            self.enc_buf += self.add_transition('ArduinoJson::JsonObject& {} = {}.createNestedObject("{}");\n'.format(
                child, enc_node, key), [len(key)])
            self.children.add(child)
            self.from_json(value, child, '{}["{}"]'.format(dec_node, key))

        elif type(value) == float:
            self.enc_buf += self.add_transition('{}["{}"] = {};\n'.format(enc_node, key, value), [len(key)])
            self.dec_buf += 'kout << {}["{}"].as<{}>();\n'.format(dec_node, key, self.float_type)
            self.assign_and_kout(self.float_type, '{}["{}"].as<{}>()'.format(dec_node, key, self.float_type), transition_args = [len(key)])

        elif type(value) == int:
            self.enc_buf += self.add_transition('{}["{}"] = {};\n'.format(enc_node, key, value), [len(key)])
            self.dec_buf += 'kout << {}["{}"].as<{}>();\n'.format(dec_node, key, self.int_type)
            self.assign_and_kout(self.int_type, '{}["{}"].as<{}>()'.format(dec_node, key, self.int_type), transition_args = [len(key)])

        else:
            self.note_unsupported(tvalue)

    def from_json(self, data, enc_node = 'root', dec_node = 'root'):
        if type(data) == dict:
            for key in sorted(data.keys()):
                self.add_to_dict(enc_node, dec_node, key, data[key])
        elif type(data) == list:
            for i, elem in enumerate(data):
                self.add_to_list(enc_node, dec_node, i, elem)


class CapnProtoC(DummyProtocol):

    def __init__(self, data, max_serialized_bytes = 128, packed = False, trail = ['benchmark'], int_type = 'uint16_t', float_type = 'float', dec_index = 0):
        super().__init__()
        self.data = data
        self.max_serialized_bytes = max_serialized_bytes
        self.packed = packed
        self.name = trail[-1]
        self.trail = trail
        self.int_type = int_type
        self.proto_int_type = self.int_type_to_proto_type(int_type)
        self.float_type = float_type
        self.proto_float_type = self.float_type_to_proto_type(float_type)
        self.dec_index = dec_index
        self.trail_name = '_'.join(map(lambda x: x.capitalize(), trail))
        self.proto_buf = ''
        self.enc_buf += 'struct {} {};\n'.format(self.trail_name, self.name)
        self.cc_tail = ''
        self.key_counter = 0
        self.from_json(data)

    def int_type_to_proto_type(self, int_type):
        sign = ''
        if int_type[0] == 'u':
            sign = 'U'
        if '8' in int_type:
            self.int_bits = 8
            return sign + 'Int8'
        if '16' in int_type:
            self.int_bits = 16
            return sign + 'Int16'
        if '32' in int_type:
            self.int_bits = 32
            return sign + 'Int32'
        self.int_bits = 64
        return sign + 'Int64'

    def float_type_to_proto_type(self, float_type):
        if float_type == 'float':
            self.float_bits = 32
            return 'Float32'
        self.float_bits = 64
        return 'Float64'

    def is_ascii(self):
        return False

    def get_proto(self):
        return '@0xad5b236043de2389;\n\n' + self.proto_buf

    def get_extra_files(self):
        return {
            'capnp_c_bench.capnp' : self.get_proto()
        }

    def get_buffer_declaration(self):
        ret = 'uint8_t buf[{:d}];\n'.format(self.max_serialized_bytes)
        ret += 'uint16_t serialized_size;\n'
        return ret

    def get_buffer_name(self):
        return 'buf'

    def get_encode(self):
        ret = 'struct capn c;\n'
        ret += 'capn_init_malloc(&c);\n'
        ret += 'capn_ptr cr = capn_root(&c);\n'
        ret += 'struct capn_segment *cs = cr.seg;\n\n'
        ret += '{}_ptr {}_ptr = new_{}(cs);\n'.format(
            self.trail_name, self.name, self.trail_name)

        tail = 'write_{}(&{}, {}_ptr);\n'.format(
            self.trail_name, self.name, self.name)
        tail += 'capn_setp(cr, 0, {}_ptr.p);\n'.format(self.name)

        return ret + self.enc_buf + self.cc_tail + tail

    def get_serialize(self):
        ret = 'serialized_size = capn_write_mem(&c, buf, sizeof(buf), {:d});\n'.format(self.packed)
        ret += 'capn_free(&c);\n'
        return ret

    def get_deserialize(self):
        ret = 'struct capn c;\n'
        ret += 'capn_init_mem(&c, buf, serialized_size, 0);\n'
        return ret

    def get_decode_and_output(self):
        ret = '{}_ptr {}_ptr;\n'.format(self.trail_name, self.name)
        ret += '{}_ptr.p = capn_getp(capn_root(&c), 0, 1);\n'.format(self.name)
        ret += 'struct {} {};\n'.format(self.trail_name, self.name)
        ret += 'kout << dec << "dec:";\n'
        ret += self.dec_buf
        ret += 'kout << endl;\n'
        ret += 'capn_free(&c);\n'
        return ret

    def get_decode_vars(self):
        return self.dec_buf0

    def get_decode(self):
        ret = '{}_ptr {}_ptr;\n'.format(self.trail_name, self.name)
        ret += '{}_ptr.p = capn_getp(capn_root(&c), 0, 1);\n'.format(self.name)
        ret += 'struct {} {};\n'.format(self.trail_name, self.name)
        ret += self.dec_buf1
        ret += 'capn_free(&c);\n'
        return ret

    def get_decode_output(self):
        return 'kout << dec << "dec:";\n' + self.dec_buf2 + 'kout << endl;\n';

    def get_length_var(self):
        return 'serialized_size'

    def add_field(self, fieldtype, key, value):
        extra = ''
        texttype = self.proto_int_type

        if fieldtype == str:
            texttype = 'Text'
        elif fieldtype == float:
            texttype = self.proto_float_type
        elif fieldtype == dict:
            texttype = key.capitalize()

        if type(value) == list:
            texttype = 'List({})'.format(texttype)

        self.proto_buf += '{} @{:d} :{};\n'.format(
            key, self.key_counter, texttype)
        self.key_counter += 1

        if fieldtype == str:
            self.enc_buf += 'capn_text {}_text;\n'.format(key)
            self.enc_buf += '{}_text.len = {:d};\n'.format(key, len(value))
            self.enc_buf += '{}_text.str = "{}";\n'.format(key, value)
            self.enc_buf += '{}_text.seg = NULL;\n'.format(key)
            self.enc_buf += '{}.{} = {}_text;\n\n'.format(self.name, key, key)
            self.dec_buf += 'kout << {}.{}.str;\n'.format(self.name, key)
            self.assign_and_kout('char const *', '{}.{}.str'.format(self.name, key))
        elif fieldtype == dict:
            pass # content is handled recursively in add_to_dict
        elif type(value) == list:
            if type(value[0]) == float:
                self.enc_buf += self.add_transition('{}.{} = capn_new_list{:d}(cs, {:d});\n'.format(
                    self.name, key, self.float_bits, len(value)), [len(value)])
                for i, elem in enumerate(value):
                    self.enc_buf += 'capn_set{:d}({}.{}, {:d}, capn_from_f{:d}({:f}));\n'.format(
                        self.float_bits, self.name, key, i, self.float_bits, elem)
                    self.dec_buf += 'kout << capn_to_f{:d}(capn_get{:d}({}.{}, {:d}));\n'.format(
                        self.float_bits, self.float_bits, self.name, key, i)
                    self.assign_and_kout(self.float_type, 'capn_to_f{:d}(capn_get{:d}({}.{}, {:d}))'.format(self.float_bits, self.float_bits, self.name, key, i))
            else:
                self.enc_buf += self.add_transition('{}.{} = capn_new_list{:d}(cs, {:d});\n'.format(
                    self.name, key, self.int_bits, len(value)), [len(value)])
                for i, elem in enumerate(value):
                    self.enc_buf += 'capn_set{:d}({}.{}, {:d}, {:d});\n'.format(
                        self.int_bits, self.name, key, i, elem)
                    self.dec_buf += 'kout << capn_get{:d}({}.{}, {:d});\n'.format(
                        self.int_bits, self.name, key, i)
                    self.assign_and_kout(self.int_type, 'capn_get{:d}({}.{}, {:d})'.format(self.int_bits, self.name, key, i))
        elif fieldtype == float:
            self.enc_buf += '{}.{} = {};\n\n'.format(self.name, key, value)
            self.dec_buf += 'kout << {}.{};\n'.format(self.name, key)
            self.assign_and_kout(self.float_type, '{}.{}'.format(self.name, key))

        elif fieldtype == int:
            self.enc_buf += '{}.{} = {};\n\n'.format(self.name, key, value)
            self.dec_buf += 'kout << {}.{};\n'.format(self.name, key)
            self.assign_and_kout(self.int_type, '{}.{}'.format(self.name, key))

        else:
            self.note_unsupported(value)

    def add_to_dict(self, key, value):
        if type(value) == str:
            if len(value) and value[0] == '$':
                self.add_field(int, key, value[1:])
            else:
                self.add_field(str, key, value)
        elif type(value) == list:
            self.add_field(type(value[0]), key, value)
        elif type(value) == dict:
            trail = list(self.trail)
            trail.append(key)
            nested = CapnProtoC(value, trail = trail, int_type = self.int_type, float_type = self.float_type, dec_index = self.dec_index)
            self.add_field(dict, key, value)
            self.enc_buf += '{}.{} = new_{}_{}(cs);\n'.format(
                self.name, key, self.trail_name, key.capitalize())
            self.enc_buf += nested.enc_buf
            self.enc_buf += 'write_{}_{}(&{}, {}.{});\n'.format(
                self.trail_name, key.capitalize(), key, self.name, key)
            self.dec_buf += 'struct {}_{} {};\n'.format(self.trail_name, key.capitalize(), key)
            self.dec_buf += 'read_{}_{}(&{}, {}.{});\n'.format(self.trail_name, key.capitalize(), key, self.name, key)
            self.dec_buf += nested.dec_buf
            self.dec_buf0 += nested.dec_buf0
            self.dec_buf1 += 'struct {}_{} {};\n'.format(self.trail_name, key.capitalize(), key)
            self.dec_buf1 += 'read_{}_{}(&{}, {}.{});\n'.format(self.trail_name, key.capitalize(), key, self.name, key)
            self.dec_buf1 += nested.dec_buf1
            self.dec_buf2 += nested.dec_buf2
            self.dec_index = nested.dec_index
            self.proto_buf += nested.proto_buf
        else:
            self.add_field(type(value), key, value)

    def from_json(self, data):
        self.proto_buf += 'struct {} {{\n'.format(self.name.capitalize())
        if type(data) == dict:
            for key in sorted(data.keys()):
                self.add_to_dict(key, data[key])
        self.proto_buf += '}\n'

class ManualJSON(DummyProtocol):

    def __init__(self, data):
        super().__init__()
        self.data = data
        self.max_serialized_bytes = self.get_serialized_length() + 2
        self.buf = 'BufferOutput<> bout(buf);\n'
        self.buf += 'bout << "{";\n'
        self.from_json(data)
        self.buf += 'bout << "}";\n'

    def get_serialized_length(self):
        return len(json.dumps(self.data))

    def can_get_serialized_length(self):
        return True

    def is_ascii(self):
        return True

    def get_buffer_declaration(self):
        return 'char buf[{:d}];\n'.format(self.max_serialized_bytes);

    def get_buffer_name(self):
        return 'buf'

    def get_encode(self):
        return self.buf

    def get_length_var(self):
        return 'bout.size()'

    def add_to_list(self, value, is_last):
        if type(value) == str:
            if len(value) and value[0] == '$':
                self.buf += 'bout << dec << {}'.format(value[1:])
            else:
                self.buf += self.add_transition('bout << "\\"{}\\""'.format(value), [len(value)])

        elif type(value) == list:
            self.buf += 'bout << "[";\n'
            self.from_json(value)
            self.buf += 'bout << "]"'

        elif type(value) == dict:
            self.buf += 'bout << "{";\n'
            self.from_json(value)
            self.buf += 'bout << "}"'

        else:
            self.buf += 'bout << {}'.format(value)

        if is_last:
            self.buf += ';\n';
        else:
            self.buf += ' << ",";\n'

    def add_to_dict(self, key, value, is_last):
        if type(value) == str:
            if len(value) and value[0] == '$':
                self.buf += 'bout << "\\"{}\\":" << dec << {}'.format(key, value[1:])
            else:
                self.buf += self.add_transition('bout << "\\"{}\\":\\"{}\\""'.format(key, value), [len(key), len(value)])

        elif type(value) == list:
            self.buf += self.add_transition('bout << "\\"{}\\":[";\n'.format(key), [len(key)])
            self.from_json(value)
            self.buf += 'bout << "]"'

        elif type(value) == dict:
            # '{{' is an escaped '{' character
            self.buf += self.add_transition('bout << "\\"{}\\":{{";\n'.format(key), [len(key)])
            self.from_json(value)
            self.buf += 'bout << "}"'

        else:
            self.buf += self.add_transition('bout << "\\"{}\\":" << {}'.format(key, value), [len(key)])

        if is_last:
            self.buf += ';\n'
        else:
            self.buf += ' << ",";\n'

    def from_json(self, data):
        if type(data) == dict:
            keys = sorted(data.keys())
            for key in keys:
                self.add_to_dict(key, data[key], key == keys[-1])
        elif type(data) == list:
            for i, elem in enumerate(data):
                self.add_to_list(elem, i == len(data) - 1)

class ModernJSON(DummyProtocol):

    def __init__(self, data, output_format = 'json'):
        super().__init__()
        self.data = data
        self.output_format = output_format
        self.buf = 'nlohmann::json js;\n'
        self.from_json(data)

    def is_ascii(self):
        if self.output_format == 'json':
            return True
        return False

    def get_buffer_name(self):
        return 'out'

    def get_encode(self):
        return self.buf

    def get_serialize(self):
        if self.output_format == 'json':
            return 'std::string out = js.dump();\n'
        elif self.output_format == 'bson':
            return 'std::vector<std::uint8_t> out = nlohmann::json::to_bson(js);\n'
        elif self.output_format == 'cbor':
            return 'std::vector<std::uint8_t> out = nlohmann::json::to_cbor(js);\n'
        elif self.output_format == 'msgpack':
            return 'std::vector<std::uint8_t> out = nlohmann::json::to_msgpack(js);\n'
        elif self.output_format == 'ubjson':
            return 'std::vector<std::uint8_t> out = nlohmann::json::to_ubjson(js);\n'
        else:
            raise ValueError('invalid output format {}'.format(self.output_format))

    def get_serialized_length(self):
        if self.output_format == 'json':
            return len(json.dumps(self.data))
        elif self.output_format == 'bson':
            return len(bson.BSON.encode(self.data))
        elif self.output_format == 'cbor':
            return len(cbor.dumps(self.data))
        elif self.output_format == 'msgpack':
            return len(msgpack.dumps(self.data))
        elif self.output_format == 'ubjson':
            return len(ubjson.dumpb(self.data))
        else:
            raise ValueError('invalid output format {}'.format(self.output_format))

    def can_get_serialized_length(self):
        return True

    def get_length_var(self):
        return 'out.size()'

    def add_to_list(self, prefix, index, value):
        if type(value) == str:
            if len(value) and value[0] == '$':
                self.buf += value[1:]
                self.buf += '{}[{:d}] = {};\n'.format(prefix, index, value[1:])
            else:
                self.buf += '{}[{:d}] = "{}";\n'.format(prefix, index, value)

        else:
            self.buf += '{}[{:d}] = {};\n'.format(prefix, index, value)

    def add_to_dict(self, prefix, key, value):
        if type(value) == str:
            if len(value) and value[0] == '$':
                self.buf += '{}["{}"] = {};\n'.format(prefix, key, value[1:])
            else:
                self.buf += '{}["{}"] = "{}";\n'.format(prefix, key, value)

        elif type(value) == list:
            self.from_json(value, '{}["{}"]'.format(prefix, key))

        elif type(value) == dict:
            self.from_json(value, '{}["{}"]'.format(prefix, key))

        else:
            self.buf += '{}["{}"] = {};\n'.format(prefix, key, value)

    def from_json(self, data, prefix = 'js'):
        if type(data) == dict:
            for key in sorted(data.keys()):
                self.add_to_dict(prefix, key, data[key])
        elif type(data) == list:
            for i, elem in enumerate(data):
                self.add_to_list(prefix, i, elem)

class MPack(DummyProtocol):

    def __init__(self, data, int_type = 'uint16_t', float_type = 'float'):
        super().__init__()
        self.data = data
        self.max_serialized_bytes = self.get_serialized_length() + 2
        self.int_type = int_type
        self.float_type = float_type
        self.enc_buf += 'mpack_writer_t writer;\n'
        self.enc_buf += self.add_transition('mpack_writer_init(&writer, buf, sizeof(buf));\n', [self.max_serialized_bytes])
        self.dec_buf0 += 'char strbuf[16];\n'
        self.from_json(data)

    def get_serialized_length(self):
        return len(msgpack.dumps(self.data))

    def can_get_serialized_length(self):
        return True

    def is_ascii(self):
        return False

    def get_buffer_declaration(self):
        ret = 'char buf[{:d}];\n'.format(self.max_serialized_bytes)
        ret += 'uint16_t serialized_size;\n'
        return ret

    def get_buffer_name(self):
        return 'buf'

    def get_encode(self):
        return self.enc_buf

    def get_serialize(self):
        ret = 'serialized_size = mpack_writer_buffer_used(&writer);\n'
        ret += 'if (mpack_writer_destroy(&writer) != mpack_ok) {\n'
        ret += 'kout << "Encoding failed" << endl;\n'
        ret += '}\n'
        return ret

    def get_deserialize(self):
        ret = 'mpack_reader_t reader;\n'
        ret += self.add_transition('mpack_reader_init_data(&reader, buf, serialized_size);\n', [self.max_serialized_bytes])
        return ret

    def get_decode_and_output(self):
        ret = 'kout << dec << "dec:";\n'
        ret += 'char strbuf[16];\n'
        return ret + self.dec_buf + 'kout << endl;\n'

    def get_decode_vars(self):
        return self.dec_buf0

    def get_decode(self):
        return self.dec_buf1

    def get_decode_output(self):
        return 'kout << dec << "dec:";\n' + self.dec_buf2 + 'kout << endl;\n';

    def get_length_var(self):
        return 'serialized_size'

    def add_value(self, value):
        if type(value) == str:
            if len(value) and value[0] == '$':
                self.enc_buf += 'mpack_write(&writer, {});\n'.format(value[1:])
                self.dec_buf += 'kout << mpack_expect_uint(&reader);\n'
                self.assign_and_kout(self.int_type, 'mpack_expect_uint(&reader)')
            else:
                self.enc_buf += self.add_transition('mpack_write_cstr_or_nil(&writer, "{}");\n'.format(value), [len(value)])
                self.dec_buf += 'mpack_expect_cstr(&reader, strbuf, sizeof(strbuf));\n'
                self.dec_buf += 'kout << strbuf;\n'
                self.dec_buf1 += self.add_transition('mpack_expect_cstr(&reader, strbuf, sizeof(strbuf));\n', [len(value)])
                self.dec_buf2 += 'kout << strbuf;\n'
        elif type(value) == list:
            self.from_json(value)
        elif type(value) == dict:
            self.from_json(value)
        elif type(value) == int:
            self.enc_buf += 'mpack_write(&writer, ({}){:d});\n'.format(self.int_type, value)
            self.dec_buf += 'kout << mpack_expect_uint(&reader);\n'
            self.assign_and_kout(self.int_type, 'mpack_expect_uint(&reader)')
        elif type(value) == float:
            self.enc_buf += 'mpack_write(&writer, ({}){:f});\n'.format(self.float_type, value)
            self.dec_buf += 'kout << mpack_expect_float(&reader);\n'
            self.assign_and_kout(self.float_type, 'mpack_expect_float(&reader)')
        else:
            self.note_unsupported(value)

    def from_json(self, data):
        if type(data) == dict:
            self.enc_buf += self.add_transition('mpack_start_map(&writer, {:d});\n'.format(len(data)), [len(data)])
            self.dec_buf += 'mpack_expect_map_max(&reader, {:d});\n'.format(len(data))
            self.dec_buf1 += self.add_transition('mpack_expect_map_max(&reader, {:d});\n'.format(len(data)), [len(data)])
            for key in sorted(data.keys()):
                self.enc_buf += self.add_transition('mpack_write_cstr(&writer, "{}");\n'.format(key), [len(key)])
                self.dec_buf += 'mpack_expect_cstr(&reader, strbuf, sizeof(strbuf));\n'
                self.dec_buf1 += self.add_transition('mpack_expect_cstr(&reader, strbuf, sizeof(strbuf));\n', [len(key)])
                self.add_value(data[key])
            self.enc_buf += 'mpack_finish_map(&writer);\n'
            self.dec_buf += 'mpack_done_map(&reader);\n'
            self.dec_buf1 += 'mpack_done_map(&reader);\n'
        if type(data) == list:
            self.enc_buf += self.add_transition('mpack_start_array(&writer, {:d});\n'.format(len(data)), [len(data)])
            self.dec_buf += 'mpack_expect_array_max(&reader, {:d});\n'.format(len(data))
            self.dec_buf1 += self.add_transition('mpack_expect_array_max(&reader, {:d});\n'.format(len(data)), [len(data)])
            for elem in data:
                self.add_value(elem);
            self.enc_buf += 'mpack_finish_array(&writer);\n'
            self.dec_buf += 'mpack_done_array(&reader);\n'
            self.dec_buf1 += 'mpack_done_array(&reader);\n'

class NanoPB(DummyProtocol):

    def __init__(self, data, max_serialized_bytes = 256, cardinality = 'required', use_maps = False, max_string_length = None, cc_prefix = '', name = 'Benchmark',
        int_type = 'uint16_t', float_type = 'float', dec_index = 0):
        super().__init__()
        self.data = data
        self.max_serialized_bytes = max_serialized_bytes
        self.cardinality = cardinality
        self.use_maps = use_maps
        self.max_strlen = max_string_length
        self.cc_prefix = cc_prefix
        self.name = name
        self.int_type = int_type
        self.proto_int_type = self.int_type_to_proto_type(int_type)
        self.float_type = float_type
        self.proto_float_type = self.float_type_to_proto_type(float_type)
        self.dec_index = dec_index
        self.fieldnum = 1
        self.proto_head = 'syntax = "proto2";\nimport "src/app/prototest/nanopb.proto";\n\n'
        self.proto_fields = ''
        self.proto_options = ''
        self.sub_protos = []
        self.cc_encoders = ''
        self.from_json(data)

    def is_ascii(self):
        return False

    def int_type_to_proto_type(self, int_type):
        sign = 'u'
        if int_type[0] != 'u':
            sign = ''
        if '64' in int_type:
            self.int_bits = 64
            return sign + 'int64'
        # Protocol Buffers only have 32 and 64 bit integers, so we default to 32
        self.int_bits = 32
        return sign + 'int32'

    def float_type_to_proto_type(self, float_type):
        if float_type == 'float':
            self.float_bits = 32
        else:
            self.float_bits = 64
        return float_type

    def get_buffer_declaration(self):
        ret = 'uint8_t buf[{:d}];\n'.format(self.max_serialized_bytes)
        ret += 'uint16_t serialized_size;\n'
        return ret + self.get_cc_functions()

    def get_buffer_name(self):
        return 'buf'

    def get_serialize(self):
        ret = 'pb_ostream_t stream = pb_ostream_from_buffer(buf, sizeof(buf));\n'
        ret += 'pb_encode(&stream, Benchmark_fields, &msg);\n'
        ret += 'serialized_size = stream.bytes_written;\n'
        return ret

    def get_deserialize(self):
        ret = 'Benchmark msg = Benchmark_init_zero;\n'
        ret += 'pb_istream_t stream = pb_istream_from_buffer(buf, serialized_size);\n'
        ret += 'if (pb_decode(&stream, Benchmark_fields, &msg) == false) {\n'
        ret += 'kout << "deserialized failed" << endl;\n'
        ret += '}\n'
        return ret

    def get_decode_and_output(self):
        return 'kout << dec << "dec:";\n' + self.dec_buf + 'kout << endl;\n'

    def get_decode_vars(self):
        return self.dec_buf0

    def get_decode(self):
        return self.dec_buf1

    def get_decode_output(self):
        return 'kout << dec << "dec:";\n' + self.dec_buf2 + 'kout << endl;\n';

    def get_length_var(self):
        return 'serialized_size'

    def add_field(self, cardinality, fieldtype, key, value):
        extra = ''
        texttype = self.proto_int_type
        dectype = self.int_type
        if fieldtype == str:
            texttype = 'string'
        elif fieldtype == float:
            texttype = self.proto_float_type
            dectype = self.float_type
        elif fieldtype == dict:
            texttype = key.capitalize()
        if type(value) == list:
            extra = '[(nanopb).max_count = {:d}]'.format(len(value))
            self.enc_buf += 'msg.{}{}_count = {:d};\n'.format(self.cc_prefix, key, len(value))
        self.proto_fields += '{} {} {} = {:d} {};\n'.format(
            cardinality, texttype, key, self.fieldnum, extra)
        self.fieldnum += 1
        if fieldtype == str:
            if cardinality == 'optional':
                self.enc_buf += 'msg.{}has_{} = true;\n'.format(self.cc_prefix, key)
            if self.max_strlen:
                self.proto_options += '{}.{} max_size:{:d}\n'.format(self.name, key, self.max_strlen)
                i = -1
                for i, character in enumerate(value):
                    self.enc_buf += '''msg.{}{}[{:d}] = '{}';\n'''.format(self.cc_prefix, key, i, character)
                self.enc_buf += 'msg.{}{}[{:d}] = 0;\n'.format(self.cc_prefix, key, i+1)
                self.dec_buf += 'kout << msg.{}{};\n'.format(self.cc_prefix, key)
                self.assign_and_kout('char *', 'msg.{}{}'.format(self.cc_prefix, key))
            else:
                self.cc_encoders += 'bool encode_{}(pb_ostream_t *stream, const pb_field_t *field, void * const *arg)\n'.format(key)
                self.cc_encoders += '{\n'
                self.cc_encoders += 'if (!pb_encode_tag_for_field(stream, field)) return false;\n'
                self.cc_encoders += 'return pb_encode_string(stream, (uint8_t*)"{}", {:d});\n'.format(value, len(value))
                self.cc_encoders += '}\n'
                self.enc_buf += 'msg.{}{}.funcs.encode = encode_{};\n'.format(self.cc_prefix, key, key)
                self.dec_buf += '// TODO decode string {}{} via callback\n'.format(self.cc_prefix, key)
                self.dec_buf1 += '// TODO decode string {}{} via callback\n'.format(self.cc_prefix, key)
        elif fieldtype == dict:
            if cardinality == 'optional':
                self.enc_buf += 'msg.{}has_{} = true;\n'.format(self.cc_prefix, key)
            # The rest is handled recursively in add_to_dict
        elif type(value) == list:
            for i, elem in enumerate(value):
                self.enc_buf += 'msg.{}{}[{:d}] = {};\n'.format(self.cc_prefix, key, i, elem)
                self.dec_buf += 'kout << msg.{}{}[{:d}];\n'.format(self.cc_prefix, key, i)
                if fieldtype == float:
                    self.assign_and_kout(self.float_type, 'msg.{}{}[{:d}]'.format(self.cc_prefix, key, i))
                elif fieldtype == int:
                    self.assign_and_kout(self.int_type, 'msg.{}{}[{:d}]'.format(self.cc_prefix, key, i))
        elif fieldtype == int:
            if cardinality == 'optional':
                self.enc_buf += 'msg.{}has_{} = true;\n'.format(self.cc_prefix, key)
            self.enc_buf += 'msg.{}{} = {};\n'.format(self.cc_prefix, key, value)
            self.dec_buf += 'kout << msg.{}{};\n'.format(self.cc_prefix, key)
            self.assign_and_kout(self.int_type, 'msg.{}{}'.format(self.cc_prefix, key))
        elif fieldtype == dict:
            if cardinality == 'optional':
                self.enc_buf += 'msg.{}has_{} = true;\n'.format(self.cc_prefix, key)
            self.enc_buf += 'msg.{}{} = {};\n'.format(self.cc_prefix, key, value)
            self.dec_buf += 'kout << msg.{}{};\n'.format(self.cc_prefix, key)
            self.assign_and_kout(self.float_type, 'msg.{}{}'.format(self.cc_prefix, key))
        else:
            self.note_unsupported(value)

    def get_proto(self):
        return self.proto_head + '\n\n'.join(self.get_message_definitions('Benchmark'))

    def get_proto_options(self):
        return self.proto_options

    def get_extra_files(self):
        return {
            'nanopbbench.proto' : self.get_proto(),
            'nanopbbench.options' : self.get_proto_options()
        }

    def get_message_definitions(self, msgname):
        ret = list(self.sub_protos)
        ret.append('message {} {{\n'.format(msgname) + self.proto_fields + '}\n')
        return ret

    def get_encode(self):
        ret = 'Benchmark msg = Benchmark_init_zero;\n'
        return ret + self.enc_buf

    def get_cc_functions(self):
        return self.cc_encoders

    def add_to_dict(self, key, value):
        if type(value) == str:
            if len(value) and value[0] == '$':
                self.add_field(self.cardinality, int, key, value[1:])
            else:
                self.add_field(self.cardinality, str, key, value)
        elif type(value) == list:
            self.add_field('repeated', type(value[0]), key, value)
        elif type(value) == dict:
            nested_proto = NanoPB(
                value, max_string_length = self.max_strlen, cardinality = self.cardinality, use_maps = self.use_maps, cc_prefix =
                '{}{}.'.format(self.cc_prefix, key), name = key.capitalize(),
                int_type = self.int_type, float_type = self.float_type,
                dec_index = self.dec_index)
            self.sub_protos.extend(nested_proto.get_message_definitions(key.capitalize()))
            self.proto_options += nested_proto.proto_options
            self.cc_encoders += nested_proto.cc_encoders
            self.add_field(self.cardinality, dict, key, value)
            self.enc_buf += nested_proto.enc_buf
            self.dec_buf += nested_proto.dec_buf
            self.dec_buf0 += nested_proto.dec_buf0
            self.dec_buf1 += nested_proto.dec_buf1
            self.dec_buf2 += nested_proto.dec_buf2
            self.dec_index = nested_proto.dec_index
        else:
            self.add_field(self.cardinality, type(value), key, value)

    def from_json(self, data):
        if type(data) == dict:
            for key in sorted(data.keys()):
                self.add_to_dict(key, data[key])



class UBJ(DummyProtocol):

    def __init__(self, data, max_serialized_bytes = 255, int_type = 'uint16_t', float_type = 'float'):
        super().__init__()
        self.data = data
        self.max_serialized_bytes = self.get_serialized_length() + 2
        self.int_type = int_type
        self.float_type = self.parse_float_type(float_type)
        self.enc_buf += 'ubjw_context_t* ctx = ubjw_open_memory(buf, buf + sizeof(buf));\n'
        self.enc_buf += 'ubjw_begin_object(ctx, UBJ_MIXED, 0);\n'
        self.from_json('root', data)
        self.enc_buf += 'ubjw_end(ctx);\n'

    def get_serialized_length(self):
        return len(ubjson.dumpb(self.data))

    def can_get_serialized_length(self):
        return True

    def is_ascii(self):
        return False

    def parse_float_type(self, float_type):
        if float_type == 'float':
            self.float_bits = 32
        else:
            self.float_bits = 64
        return float_type

    def get_buffer_declaration(self):
        ret = 'uint8_t buf[{:d}];\n'.format(self.max_serialized_bytes)
        ret += 'uint16_t serialized_size;\n'
        return ret

    def get_buffer_name(self):
        return 'buf'

    def get_length_var(self):
        return 'serialized_size'

    def get_serialize(self):
        return 'serialized_size = ubjw_close_context(ctx);\n'

    def get_encode(self):
        return self.enc_buf

    def get_deserialize(self):
        ret = 'ubjr_context_t* ctx = ubjr_open_memory(buf, buf + serialized_size);\n'
        ret += 'ubjr_dynamic_t dynamic_root = ubjr_read_dynamic(ctx);\n'
        ret += 'ubjr_dynamic_t* root_values = (ubjr_dynamic_t*)dynamic_root.container_object.values;\n'
        return ret

    def get_decode_and_output(self):
        ret = 'kout << dec << "dec:";\n'
        ret += self.dec_buf
        ret += 'kout << endl;\n'
        ret += 'ubjr_cleanup_dynamic(&dynamic_root);\n' # This causes the data (including all strings) to be free'd
        ret += 'ubjr_close_context(ctx);\n'
        return ret

    def get_decode_vars(self):
        return self.dec_buf0

    def get_decode(self):
        return self.dec_buf1

    def get_decode_output(self):
        ret =  'kout << dec << "dec:";\n' + self.dec_buf2 + 'kout << endl;\n'
        ret += 'ubjr_cleanup_dynamic(&dynamic_root);\n'
        ret += 'ubjr_close_context(ctx);\n'
        return ret

    def add_to_list(self, root, index, value):
        if type(value) == str:
            if len(value) and value[0] == '$':
                self.enc_buf += 'ubjw_write_integer(ctx, {});\n'.format(value[1:])
                self.dec_buf += 'kout << {}_values[{:d}].integer;\n'.format(root, index)
                self.assign_and_kout(self.int_type, '{}_values[{:d}].integer'.format(root, index))
            else:
                self.enc_buf += self.add_transition('ubjw_write_string(ctx, "{}");\n'.format(value), [len(value)])
                self.dec_buf += 'kout << {}_values[{:d}].string;\n'.format(root, index)
                self.assign_and_kout('char *', '{}_values[{:d}].string'.format(root, index))

        elif type(value) == list:
            self.enc_buf += 'ubjw_begin_array(ctx, UBJ_MIXED, 0);\n'.format(value)
            self.dec_buf += '// decoding nested lists is not supported\n'
            self.dec_buf1 += '// decoding nested lists is not supported\n'
            self.from_json(root, value)
            self.enc_buf += 'ubjw_end(ctx);\n'

        elif type(value) == dict:
            self.enc_buf += 'ubjw_begin_object(ctx, UBJ_MIXED, 0);\n'.format(value)
            self.dec_buf += '// decoding objects in lists is not supported\n'
            self.dec_buf1 += '// decoding objects in lists is not supported\n'
            self.from_json(root, value)
            self.enc_buf += 'ubjw_end(ctx);\n'

        elif type(value) == float:
            self.enc_buf += 'ubjw_write_float{:d}(ctx, {});\n'.format(self.float_bits, value)
            self.dec_buf += 'kout << {}_values[{:d}].real;\n'.format(root, index)
            self.assign_and_kout(self.float_type, '{}_values[{:d}].real'.format(root, index))

        elif type(value) == int:
            self.enc_buf += 'ubjw_write_integer(ctx, {});\n'.format(value)
            self.dec_buf += 'kout << {}_values[{:d}].integer;\n'.format(root, index)
            self.assign_and_kout(self.int_type, '{}_values[{:d}].integer'.format(root, index))

        else:
            raise TypeError('Cannot handle {} of type {}'.format(value, type(value)))

    def add_to_dict(self, root, index, key, value):
        if type(value) == str:
            if len(value) and value[0] == '$':
                self.enc_buf += self.add_transition('ubjw_write_key(ctx, "{}"); ubjw_write_integer(ctx, {});\n'.format(key, value[1:]), [len(key)])
                self.dec_buf += 'kout << {}_values[{:d}].integer;\n'.format(root, index)
                self.assign_and_kout(self.int_type, '{}_values[{:d}].integer'.format(root, index))
            else:
                self.enc_buf += self.add_transition('ubjw_write_key(ctx, "{}"); ubjw_write_string(ctx, "{}");\n'.format(key, value), [len(key), len(value)])
                self.dec_buf += 'kout << {}_values[{:d}].string;\n'.format(root, index)
                self.assign_and_kout('char *', '{}_values[{:d}].string'.format(root, index))

        elif type(value) == list:
            self.enc_buf += self.add_transition('ubjw_write_key(ctx, "{}"); ubjw_begin_array(ctx, UBJ_MIXED, 0);\n'.format(key), [len(key)])
            self.dec_buf += 'ubjr_dynamic_t *{}_values = (ubjr_dynamic_t*){}_values[{:d}].container_array.values;\n'.format(
                key, root, index)
            self.dec_buf1 += 'ubjr_dynamic_t *{}_values = (ubjr_dynamic_t*){}_values[{:d}].container_array.values;\n'.format(
                key, root, index)
            self.from_json(key, value)
            self.enc_buf += 'ubjw_end(ctx);\n'

        elif type(value) == dict:
            self.enc_buf += self.add_transition('ubjw_write_key(ctx, "{}"); ubjw_begin_object(ctx, UBJ_MIXED, 0);\n'.format(key), [len(key)])
            self.dec_buf += 'ubjr_dynamic_t *{}_values = (ubjr_dynamic_t*){}_values[{:d}].container_object.values;\n'.format(
                key, root, index)
            self.dec_buf1 += 'ubjr_dynamic_t *{}_values = (ubjr_dynamic_t*){}_values[{:d}].container_object.values;\n'.format(
                key, root, index)
            self.from_json(key, value)
            self.enc_buf += 'ubjw_end(ctx);\n'

        elif type(value) == float:
            self.enc_buf += self.add_transition('ubjw_write_key(ctx, "{}"); ubjw_write_float{:d}(ctx, {});\n'.format(key, self.float_bits, value), [len(key)])
            self.dec_buf += 'kout << {}_values[{:d}].real;\n'.format(root, index)
            self.assign_and_kout(self.float_type, '{}_values[{:d}].real'.format(root, index))

        elif type(value) == int:
            self.enc_buf += self.add_transition('ubjw_write_key(ctx, "{}"); ubjw_write_integer(ctx, {});\n'.format(key, value), [len(key)])
            self.dec_buf += 'kout << {}_values[{:d}].integer;\n'.format(root, index)
            self.assign_and_kout(self.int_type, '{}_values[{:d}].integer'.format(root, index))

        else:
            raise TypeError('Cannot handle {} of type {}'.format(value, type(value)))

    def from_json(self, root, data):
        if type(data) == dict:
            keys = sorted(data.keys())
            for i, key in enumerate(keys):
                self.add_to_dict(root, i, key, data[key])
        elif type(data) == list:
            for i, elem in enumerate(data):
                self.add_to_list(root, i, elem)


class XDR(DummyProtocol):

    def __init__(self, data, max_serialized_bytes = 256, int_type = 'uint16_t', float_type = 'float'):
        super().__init__()
        self.data = data
        self.max_serialized_bytes = 256
        self.enc_int_type = int_type
        self.dec_int_type = self.parse_int_type(int_type)
        self.float_type = self.parse_float_type(float_type)
        self.enc_buf += 'BufferOutput<XDRStream> xdrstream(buf);\n'
        self.dec_buf += 'XDRInput xdrinput(buf);\n'
        self.dec_buf0 += 'XDRInput xdrinput(buf);\n'
        # By default, XDR does not even include a version / protocol specifier.
        # This seems rather impractical -> emulate that here.
        self.enc_buf += 'xdrstream << (uint32_t)22075;\n'
        self.dec_buf += 'char strbuf[16];\n'
        self.dec_buf += 'xdrinput.get_uint32();\n'
        self.dec_buf0 += 'char strbuf[16];\n'
        self.dec_buf0 += 'xdrinput.get_uint32();\n'
        self.from_json(data)

    def is_ascii(self):
        return False

    def parse_int_type(self, int_type):
        sign = ''
        if int_type[0] == 'u':
            sign = 'u'
        if '64' in int_type:
            self.int_bits = 64
            return sign + 'int64'
        else:
            self.int_bits = 32
            return sign + 'int32'

    def parse_float_type(self, float_type):
        if float_type == 'float':
            self.float_bits = 32
        else:
            self.float_bits = 64
        return float_type

    def get_buffer_declaration(self):
        ret = 'uint16_t serialized_size;\n'
        ret += 'char buf[{:d}];\n'.format(self.max_serialized_bytes)
        return ret

    def get_buffer_name(self):
        return 'buf'

    def get_length_var(self):
        return 'xdrstream.size()'

    def get_encode(self):
        return self.enc_buf

    def get_decode_and_output(self):
        return 'kout << dec << "dec:";\n' + self.dec_buf + 'kout << endl;\n'

    def get_decode_vars(self):
        return self.dec_buf0

    def get_decode(self):
        return self.dec_buf1

    def get_decode_output(self):
        return 'kout << dec << "dec:";\n' + self.dec_buf2 + 'kout << endl;\n';

    def from_json(self, data):
        if type(data) == dict:
            for key in sorted(data.keys()):
                self.from_json(data[key])
        elif type(data) == list:
            #self.enc_buf += 'xdrstream.setNextArrayLen({});\n'.format(len(data))
            #self.enc_buf += 'xdrstream << variable;\n'
            self.enc_buf += 'xdrstream << (uint32_t){:d};\n'.format(len(data))
            self.dec_buf += 'xdrinput.get_uint32();\n'
            self.dec_buf1 += 'xdrinput.get_uint32();\n'
            for elem in data:
                self.from_json(elem)
        elif type(data) == str:
            if len(data) and data[0] == '$':
                self.enc_buf += 'xdrstream << ({}){};\n'.format(self.enc_int_type, data[1:])
                self.dec_buf += 'kout << xdrinput.get_{}();\n'.format(self.dec_int_type)
                self.dec_buf0 += '{} dec_{};\n'.format(self.enc_int_type, self.dec_index)
                self.dec_buf1 += 'dec_{} = xdrinput.get_{}();;\n'.format(self.dec_index, self.dec_int_type)
                self.dec_buf2 += 'kout << dec_{};\n'.format(self.dec_index)
            else:
                # Kodierte Strings haben nicht immer ein Nullbyte am Ende
                self.enc_buf += 'xdrstream.setNextArrayLen({});\n'.format(len(data))
                self.enc_buf += self.add_transition('xdrstream << variable << "{}";\n'.format(data), [len(data)])
                self.dec_buf += 'xdrinput.get_string(strbuf);\n'
                self.dec_buf += 'kout << strbuf;\n'
                self.dec_buf1 += 'xdrinput.get_string(strbuf);\n'.format(self.dec_index)
                self.dec_buf2 += 'kout << strbuf;\n'.format(self.dec_index)
        elif type(data) == float:
            self.enc_buf += 'xdrstream << ({}){};\n'.format(self.float_type, data)
            self.dec_buf += 'kout << xdrinput.get_{}();\n'.format(self.float_type)
            self.dec_buf0 += '{} dec_{};\n'.format(self.float_type, self.dec_index)
            self.dec_buf1 += 'dec_{} = xdrinput.get_{}();\n'.format(self.dec_index, self.float_type)
            self.dec_buf2 += 'kout << dec_{};\n'.format(self.dec_index)
        elif type(data) == int:
            self.enc_buf += 'xdrstream << ({}){};\n'.format(self.enc_int_type, data)
            self.dec_buf += 'kout << xdrinput.get_{}();\n'.format(self.dec_int_type)
            self.dec_buf0 += '{} dec_{};\n'.format(self.enc_int_type, self.dec_index)
            self.dec_buf1 += 'dec_{} = xdrinput.get_{}();\n'.format(self.dec_index, self.dec_int_type)
            self.dec_buf2 += 'kout << dec_{};\n'.format(self.dec_index)
        else:
            self.enc_buf += 'xdrstream << {};\n'.format(data)
            self.dec_buf += '// unsupported type {} of {}\n'.format(type(data), data)
            self.dec_buf1 += '// unsupported type {} of {}\n'.format(type(data), data)
        self.dec_index += 1;

class Benchmark:

    def __init__(self, logfile):
        self.atomic = True
        self.logfile = logfile

    def __enter__(self):
        self.atomic = False
        with FileLock(self.logfile + '.lock'):
            if os.path.exists(self.logfile):
                with open(self.logfile, 'rb') as f:
                    self.data = ubjson.load(f)
            else:
                self.data = {}
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        with FileLock(self.logfile + '.lock'):
            with open(self.logfile, 'wb') as f:
                ubjson.dump(self.data, f)

    def _add_log_entry(self, benchmark_data, arch, libkey, bench_name, bench_index, data, key, value, error):
        if not libkey in benchmark_data:
            benchmark_data[libkey] = dict()
        if not bench_name in benchmark_data[libkey]:
            benchmark_data[libkey][bench_name] = dict()
        if not bench_index in benchmark_data[libkey][bench_name]:
            benchmark_data[libkey][bench_name][bench_index] = dict()
        this_result = benchmark_data[libkey][bench_name][bench_index]
        # data is unset for log(...) calls from postprocessing
        if data != None:
            this_result['data'] = data
        if value != None:
            this_result[key] = {
                'v' : value,
                'ts' : int(time.time())
            }
            print('{} {} {} ({}) :: {} -> {}'.format(
                libkey, bench_name, bench_index, data, key, value))
        else:
            this_result[key] = {
                'e' : error,
                'ts' : int(time.time())
            }
            print('{} {} {} ({}) :: {} -> [E] {}'.format(
                libkey, bench_name, bench_index, data, key, error))

    def log(self, arch, library, library_options, bench_name, bench_index, data, key, value = None, error = None):
        if not library_options:
            library_options = []
        libkey = '{}:{}:{}'.format(arch, library, ','.join(library_options))
        # JSON does not differentiate between int and str keys -> always use
        # str(bench_index)
        bench_index = str(bench_index)
        if self.atomic:
            with FileLock(self.logfile + '.lock'):
                if os.path.exists(self.logfile):
                    with open(self.logfile, 'rb') as f:
                        benchmark_data = ubjson.load(f)
                else:
                    benchmark_data = {}
                self._add_log_entry(benchmark_data, arch, libkey, bench_name, bench_index, data, key, value, error)
                with open(self.logfile, 'wb') as f:
                    ubjson.dump(benchmark_data, f)
        else:
            self._add_log_entry(self.data, arch, libkey, bench_name, bench_index, data, key, value, error)

    def get_snapshot(self):
        with FileLock(self.logfile + '.lock'):
            if os.path.exists(self.logfile):
                with open(self.logfile, 'rb') as f:
                    benchmark_data = ubjson.load(f)
            else:
                benchmark_data = {}
        return benchmark_data

def codegen_for_lib(library, library_options, data):
    if library == 'arduinojson':
        return ArduinoJSON(data, bufsize = 512)

    if library == 'capnproto_c':
        packed = bool(int(library_options[0]))
        return CapnProtoC(data, packed = packed)

    if library == 'manualjson':
        return ManualJSON(data)

    if library == 'modernjson':
        dataformat, = library_options
        return ModernJSON(data, dataformat)

    if library == 'mpack':
        return MPack(data)

    if library == 'nanopb':
        cardinality, strbuf = library_options
        if not len(strbuf) or strbuf == '0':
            strbuf = None
        else:
            strbuf = int(strbuf)
        return NanoPB(data, cardinality = cardinality, max_string_length = strbuf)

    if library == 'ubjson':
        return UBJ(data)

    if library == 'xdr':
        return XDR(data)

    raise ValueError('Unsupported library: {}'.format(library))

def shorten_call(snippet, lib = ''):
    """
    Remove literal arguments and variable names from ProtoBench function calls.

    This provides some generalization when modeling individual function
    calls, thus avoiding overfitting in AnalyticModel and the likes.
    """
    # The following adjustments are protobench-specific
    # "xdrstream << (uint16_t)123" -> "xdrstream << (uint16_t"
    if 'xdrstream << (' in snippet:
        snippet = snippet.split(')')[0]
    # "xdrstream << variable << ..." -> "xdrstream << variable"
    elif 'xdrstream << variable' in snippet:
        snippet = '<<'.join(snippet.split('<<')[0:2])
    elif 'xdrstream.setNextArrayLen(' in snippet:
        snippet = 'xdrstream.setNextArrayLen'
    elif 'ubjw' in snippet:
        snippet = re.sub('ubjw_write_key\(ctx, [^)]+\)', 'ubjw_write_key(ctx, ?)', snippet)
        snippet = re.sub('ubjw_write_([^(]+)\(ctx, [^)]+\)', 'ubjw_write_\\1(ctx, ?)', snippet)
    # mpack_write(&writer, (type)value) -> mpack_write(&writer, (type
    elif 'mpack_write(' in snippet:
        snippet = snippet.split(')')[0]
    # mpack_write_cstr(&writer, "foo") -> mpack_write_cstr(&writer,
    # same for mpack_write_cstr_or_nil
    elif 'mpack_write_cstr' in snippet:
        snippet = snippet.split('"')[0]
    # mpack_start_map(&writer, x) -> mpack_start_map(&writer
    # mpack_start_array(&writer, x) -> mpack_start_array(&writer
    elif 'mpack_start_' in snippet:
        snippet = snippet.split(',')[0]
    elif 'bout <<' in snippet:
        snippet = 'bout'
    elif 'msg.' in snippet:
        snippet = re.sub('msg.(?:[^[]+)(?:\[.*?\])? = .*', 'msg.? = ?', snippet)
    elif lib == 'arduinojson:':
        snippet = re.sub('ArduinoJson::JsonObject& [^ ]+ = [^.]+.createNestedObject\([^)]*\);', 'ArduinoJson::JsonObject& ? = ?.createNestedObject(?);', snippet)
        snippet = re.sub('ArduinoJson::JsonArray& [^ ]+ = [^.]+.createNestedArray\([^)]*\);', 'ArduinoJson::JsonArray& ? = ?.createNestedArray(?);', snippet)
        snippet = re.sub('root[^[]*\["[^"]*"\] = [^;]+', 'root?["?"] = ?', snippet)
        snippet = re.sub('rootl.add\([^)]*\)', 'rootl.add(?)', snippet)

    snippet = re.sub('^dec_[^ ]*', 'dec_?', snippet)
    if lib == 'arduinojson:':
        snippet = re.sub('root[^. ]+\.as', 'root[?].as', snippet)
    elif 'nanopb:' in lib:
        snippet = re.sub('= msg\.[^;]+;', '= msg.?;', snippet)
    elif lib == 'mpack:':
        snippet = re.sub('mpack_expect_([^_]+)_max\(&reader, [^)]+\)', 'mpack_expect_\\1_max(&reader, ?)', snippet)
    elif lib == 'ubjson:':
        snippet = re.sub('[^_ ]+_values[^.]+\.', '?_values[?].', snippet)

    return snippet
