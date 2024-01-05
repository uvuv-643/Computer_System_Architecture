from __future__ import annotations

import json
from enum import Enum


class OpcodeParamType(str, Enum):
    CONST = "const"
    ADDR = "addr"
    UNDEFINED = "undefined"


class OpcodeParam:
    def __init__(self, param_type: OpcodeParamType, value: any):
        self.param_type = param_type
        self.value = value

    def __str__(self):
        return f"({self.param_type}, {self.value})"


class OpcodeType(str, Enum):
    DROP = "drop"
    MUL = "mul"
    DIV = "div"
    SUB = "sub"
    ADD = "add"
    MOD = "mod"
    SWAP = "swap"
    OVER = "over"
    DUP = "dup"
    EQ = "eq"
    GR = "gr"
    LS = "ls"
    DI = "di"
    EI = "ei"
    OMIT = "omit"
    WRITE = "write"
    READ = "read"

    # not used in source code, compile-generated
    STORE = "store"
    LOAD = "load"
    PUSH = "push"
    RPOP = "rpop"  # move from return stack to data stack
    POP = "pop"  # move from data stack to return stack
    MOV = "mov"  # move to data memory from data stack
    JMP = "jmp"
    ZJMP = "zjmp"
    CALL = "call"
    RET = "ret"
    HALT = "halt"

    def __str__(self):
        return str(self.value)


class Opcode:
    def __init__(self, opcode_type: OpcodeType, params: list[OpcodeParam]):
        self.opcode_type = opcode_type
        self.params = params


class TermType(Enum):
    (
        # Term --> Opcode
        DI,
        EI,
        DUP,
        ADD,
        SUB,
        MUL,
        DIV,
        MOD,
        OMIT,
        SWAP,
        DROP,
        OVER,
        EQ,
        LS,
        GR,
        READ,
        # Term !-> Opcode
        VARIABLE,
        ALLOT,
        STORE,
        LOAD,
        IF,
        ELSE,
        THEN,
        PRINT,
        DEF,
        RET,
        DEF_INTR,
        DO,
        LOOP,
        BEGIN,
        UNTIL,
        LOOP_CNT,
        CALL,
        STRING,
    ) = range(34)


class Term:
    def __init__(self, word_number: int, term_type: TermType | None, word: str):
        self.converted = False
        self.operand = None
        self.word_number = word_number
        self.term_type = term_type
        self.word = word


def write_code(filename, code):
    with open(filename, "w", encoding="utf-8") as file:
        buf = []
        for instr in code:
            buf.append(json.dumps(instr))
        file.write("[" + ",\n ".join(buf) + "]")


def read_code(filename):
    with open(filename, encoding="utf-8") as file:
        code = json.loads(file.read())
    for instr in code:
        instr["opcode"] = OpcodeType(instr["opcode"])
        if "term" in instr:
            assert len(instr["term"]) == 3
            instr["term"] = Term(instr["term"][0], instr["term"][1], instr["term"][2])
    return code
