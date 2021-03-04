# Based on https://rosettacode.org/wiki/Modular_arithmetic#Python
# Licensed under GFDL 1.2 https://www.gnu.org/licenses/old-licenses/fdl-1.2.html
import operator
import functools


@functools.total_ordering
class Mod:
    """A class for modular arithmetic, useful to simulate behaviour of uint8 and other limited data types.

    Does not support negative values, therefore it cannot be used to emulate signed integers.

    Overloads a==b, a<b, a>b, a+b, a-b, a*b, a**b, and -a

    :param val: stored integer value
    Param mod: modulus
    """

    __slots__ = ["val", "mod"]

    def __init__(self, val, mod):
        if isinstance(val, Mod):
            val = val.val
        if not isinstance(val, int):
            raise ValueError("Value must be integer")
        if not isinstance(mod, int) or mod <= 0:
            raise ValueError("Modulo must be positive integer")
        self.val = val % mod
        self.mod = mod

    def __repr__(self):
        return "Mod({}, {})".format(self.val, self.mod)

    def __int__(self):
        return self.val

    def __eq__(self, other):
        if isinstance(other, Mod):
            self.val == other.val
        elif isinstance(other, int):
            return self.val == other
        else:
            return NotImplemented

    def __lt__(self, other):
        if isinstance(other, Mod):
            return self.val < other.val
        elif isinstance(other, int):
            return self.val < other
        else:
            return NotImplemented

    def _check_operand(self, other):
        if not isinstance(other, (int, Mod)):
            raise TypeError("Only integer and Mod operands are supported")

    def __pow__(self, other):
        self._check_operand(other)
        # We use the built-in modular exponentiation function, this way we can avoid working with huge numbers.
        return __class__(pow(self.val, int(other), self.mod), self.mod)

    def __neg__(self):
        return Mod(self.mod - self.val, self.mod)

    def __pos__(self):
        return self  # The unary plus operator does nothing.

    def __abs__(self):
        # The value is always kept non-negative, so the abs function should do nothing.
        return self


# Helper functions to build common operands based on a template.
# They need to be implemented as functions for the closures to work properly.
def _make_op(opname):
    op_fun = getattr(
        operator, opname
    )  # Fetch the operator by name from the operator module

    def op(self, other):
        self._check_operand(other)
        return Mod(op_fun(self.val, int(other)) % self.mod, self.mod)

    return op


def _make_reflected_op(opname):
    op_fun = getattr(operator, opname)

    def op(self, other):
        self._check_operand(other)
        return Mod(op_fun(int(other), self.val) % self.mod, self.mod)

    return op


# Build the actual operator overload methods based on the template.
for opname, reflected_opname in [
    ("__add__", "__radd__"),
    ("__sub__", "__rsub__"),
    ("__mul__", "__rmul__"),
]:
    setattr(Mod, opname, _make_op(opname))
    setattr(Mod, reflected_opname, _make_reflected_op(opname))


class Uint8(Mod):
    __slots__ = []

    def __init__(self, val):
        super().__init__(val, 256)

    def __repr__(self):
        return "Uint8({})".format(self.val)


class Uint16(Mod):
    __slots__ = []

    def __init__(self, val):
        super().__init__(val, 65536)

    def __repr__(self):
        return "Uint16({})".format(self.val)


class Uint32(Mod):
    __slots__ = []

    def __init__(self, val):
        super().__init__(val, 4294967296)

    def __repr__(self):
        return "Uint32({})".format(self.val)


class Uint64(Mod):
    __slots__ = []

    def __init__(self, val):
        super().__init__(val, 18446744073709551616)

    def __repr__(self):
        return "Uint64({})".format(self.val)


def simulate_int_type(int_type: str) -> Mod:
    """
    Return `Mod` subclass for given `int_type`

    :param int_type: uint8_t / uint16_t / uint32_t / uint64_t
    :returns: `Mod` subclass, e.g. Uint8
    """
    if int_type == "uint8_t":
        return Uint8
    if int_type == "uint16_t":
        return Uint16
    if int_type == "uint32_t":
        return Uint32
    if int_type == "uint64_t":
        return Uint64
    raise ValueError("unsupported integer type: {}".format(int_type))
