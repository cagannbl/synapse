from enum import Enum, auto


class Opcode(Enum):
    # Constants & Variables
    LOAD_CONST = auto()
    LOAD_NAME = auto()
    STORE_NAME = auto()
    POP_TOP = auto()

    # Binary Arithmetic & Tensor
    BINARY_ADD = auto()
    BINARY_SUB = auto()
    BINARY_MUL = auto()
    BINARY_DIV = auto()
    BINARY_FLOOR_DIV = auto()
    BINARY_MOD = auto()
    BINARY_POW = auto()
    BINARY_MATMUL = auto()       # @
    BINARY_PIPE = auto()         # |>

    # Unary Operations
    UNARY_NEGATIVE = auto()
    UNARY_NOT = auto()

    # Comparison
    COMPARE_OP = auto()          # ==, !=, <, <=, >, >=

    # Jumps & Control Flow
    JUMP = auto()
    JUMP_IF_FALSE = auto()
    JUMP_IF_TRUE = auto()

    # Functions & Execution
    MAKE_FUNCTION = auto()
    CALL_FUNCTION = auto()
    RETURN_VALUE = auto()

    # Data Structures & Member Access
    BUILD_LIST = auto()
    BUILD_DICT = auto()
    BINARY_SUBSCR = auto()
    STORE_SUBSCR = auto()
    LOAD_ATTR = auto()
    STORE_ATTR = auto()

    # AI-Native Bytecode
    BUILD_TENSOR = auto()
    DEFINE_PROMPT = auto()
    DEFINE_AGENT = auto()
    IMPORT_PYTHON = auto()
