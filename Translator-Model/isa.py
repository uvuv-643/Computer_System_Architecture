from __future__ import annotations

import json
from enum import Enum


class Opcode(str, Enum):
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
    RDUP = "rdup"
    RPOP = "rpop"
    RPUSH = "rpush"
    RADD = "radd"
    RDROP = "rdrop"
    POP = "pop"  # move from data stack to return stack
    MOV = "mov"  # move to data memory from data stack
    JMP = "jmp"
    ZJMP = "zjmp"
    RET = "ret"
    HALT = "halt"

    def __str__(self):
        return str(self.value)


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
        WRITE,
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
    ) = range(33)


class Term:
    def __init__(self, word_number: int, term_type: TermType | None, word: str):
        self.converted = False
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
        instr["opcode"] = Opcode(instr["opcode"])
        if "term" in instr:
            assert len(instr["term"]) == 3
            instr["term"] = Term(instr["term"][0], instr["term"][1], instr["term"][2])
    return code
