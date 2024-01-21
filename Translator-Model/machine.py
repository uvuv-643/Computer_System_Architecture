from __future__ import annotations

import logging
import random
import sys
import typing
from enum import Enum

import pytest as pytest
from isa import OpcodeType, read_code

logger = logging.getLogger("machine_logger")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class Selector(str, Enum):
    SP_INC = "sp_inc"
    SP_DEC = "sp_dec"
    I_INC = "i_inc"
    I_DEC = "i_dec"
    RET_STACK_PC = "ret_stack_pc"
    RET_STACK_OUT = "ret_stack_out"
    NEXT_MEM = "next_mem"
    NEXT_TOP = "next_top"
    NEXT_TEMP = "next_temp"
    TEMP_NEXT = "temp_next"
    TEMP_TOP = "temp_top"
    TEMP_RETURN = "temp_return"
    TOP_TEMP = "top_temp"
    TOP_NEXT = "top_next"
    TOP_ALU = "top_alu"
    TOP_MEM = "top_mem"
    TOP_IMMEDIATE = "top_immediate"
    TOP_INPUT = "top_input"
    PC_INC = "pc_int"
    PC_RET = "pc_ret"

    def __str__(self) -> str:
        return str(self.value)


class ALUOpcode(str, Enum):
    INC_A = "inc_a"
    INC_B = "inc_b"
    DEC_A = "dec_a"
    DEC_B = "dec_b"
    ADD = "add"
    MUL = "mul"
    DIV = "div"
    SUB = "sub"
    MOD = "mod"
    EQ = "eq"
    GR = "gr"
    LS = "ls"

    def __str__(self) -> str:
        return str(self.value)


class ALU:
    alu_operations: typing.ClassVar[list[ALUOpcode]] = [
        ALUOpcode.INC_A,
        ALUOpcode.INC_B,
        ALUOpcode.DEC_A,
        ALUOpcode.DEC_B,
        ALUOpcode.ADD,
        ALUOpcode.MUL,
        ALUOpcode.DIV,
        ALUOpcode.SUB,
        ALUOpcode.MOD,
        ALUOpcode.EQ,
        ALUOpcode.GR,
        ALUOpcode.LS,
    ]
    result = None
    src_a = None
    src_b = None
    operation = None

    def __init__(self):
        self.result = 0
        self.src_a = None
        self.src_b = None
        self.operation = None

    def calc(self) -> None:  # noqa: C901 -- function is too complex
        if self.operation == ALUOpcode.INC_A:
            self.result = self.src_a + 1
        elif self.operation == ALUOpcode.INC_B:
            self.result = self.src_b + 1
        elif self.operation == ALUOpcode.DEC_A:
            self.result = self.src_a - 1
        elif self.operation == ALUOpcode.DEC_B:
            self.result = self.src_b - 1
        elif self.operation == ALUOpcode.ADD:
            self.result = self.src_a + self.src_b
        elif self.operation == ALUOpcode.MUL:
            self.result = self.src_a * self.src_b
        elif self.operation == ALUOpcode.DIV:
            self.result = self.src_a // self.src_b
        elif self.operation == ALUOpcode.SUB:
            self.result = self.src_a - self.src_b
        elif self.operation == ALUOpcode.MOD:
            self.result = self.src_a % self.src_b
        elif self.operation == ALUOpcode.EQ:
            self.result = self.src_a == self.src_b
        elif self.operation == ALUOpcode.GR:
            self.result = self.src_a > self.src_b
        elif self.operation == ALUOpcode.LS:
            self.result = self.src_a < self.src_b
        else:
            pytest.fail(f"Unknown ALU operation: {self.operation}")

    def set_details(self, src_a, src_b, operation: ALUOpcode) -> None:
        self.src_a = src_a
        self.src_b = src_b
        self.operation = operation


class DataPath:
    memory_size = None
    memory = None
    data_stack_size = None
    data_stack = None
    return_stack_size = None
    return_stack = None

    sp = None
    i = None
    pc = None
    top = None
    next = None
    temp = None

    alu = None

    def __init__(self, memory_size: int, data_stack_size: int, return_stack_size: int):
        assert memory_size > 0, "Data memory size must be greater than zero"
        assert data_stack_size > 0, "Data stack size must be greater than zero"
        assert return_stack_size > 0, "Return stack size must be greater than zero"

        self.memory_size = memory_size
        self.data_stack_size = data_stack_size
        self.return_stack_size = return_stack_size
        self.memory = [0] * memory_size
        self.data_stack = [0] * data_stack_size
        self.return_stack = [0] * return_stack_size

        self.sp = 2
        self.i = 0
        self.pc = 0
        self.top = self.next = self.temp = 0

        self.alu = ALU()

    def signal_latch_sp(self, selector: Selector) -> None:
        if selector is Selector.SP_DEC:
            self.sp -= 1
        elif selector is Selector.SP_INC:
            self.sp += 1

    def signal_latch_i(self, selector: Selector) -> None:
        if selector is Selector.I_DEC:
            self.i -= 1
        elif selector is Selector.I_INC:
            self.i += 1

    def signal_latch_next(self, selector: Selector) -> None:
        if selector is Selector.NEXT_MEM:
            assert self.sp >= 0, "Address below 0"
            assert self.sp < self.data_stack_size, "Data stack overflow"
            self.next = self.data_stack[self.sp]
        elif selector is Selector.NEXT_TOP:
            self.next = self.top
        elif selector is Selector.NEXT_TEMP:
            self.next = self.temp

    def signal_latch_temp(self, selector: Selector) -> None:
        if selector is Selector.TEMP_RETURN:
            assert self.i >= 0, "Address below 0"
            assert self.i < self.return_stack_size, "Return stack overflow"
            self.temp = self.return_stack[self.i]
        elif selector is Selector.TEMP_TOP:
            self.temp = self.top
        elif selector is Selector.TEMP_NEXT:
            self.temp = self.next

    def signal_latch_top(self, selector: Selector, immediate=0) -> None:
        if selector is Selector.TOP_NEXT:
            self.top = self.next
        elif selector is Selector.TOP_TEMP:
            self.top = self.temp
        elif selector is Selector.TOP_INPUT:
            self.top = 47474747
        elif selector is Selector.TOP_ALU:
            self.top = self.alu.result
        elif selector is Selector.TOP_MEM:
            self.top = self.memory[self.top]
        elif selector is Selector.TOP_IMMEDIATE:
            self.top = immediate

    def signal_data_wr(self) -> None:
        assert self.sp >= 0, "Address below 0"
        assert self.sp < self.data_stack_size, "Data stack overflow"
        self.data_stack[self.sp] = self.next

    def signal_ret_wr(self, selector: Selector) -> None:
        assert self.i >= 0, "Address below 0"
        assert self.i < self.return_stack_size, "Return stack overflow"
        if selector is Selector.RET_STACK_PC:
            self.return_stack[self.i] = self.pc
        elif selector is Selector.RET_STACK_OUT:
            self.return_stack[self.i] = self.temp

    def signal_mem_wr(self) -> None:
        assert self.top >= 0, "Address below 0"
        assert self.top < self.memory_size, "Memory overflow"
        self.next = self.memory[self.top]

    def signal_alu_operation(self, operation: ALUOpcode) -> None:
        self.alu.set_details(self.top, self.next, operation)
        self.alu.calc()


def opcode_to_alu_opcode(opcode_type: OpcodeType):
    return {
        OpcodeType.MUL: ALUOpcode.MUL,
        OpcodeType.DIV: ALUOpcode.DIV,
        OpcodeType.SUB: ALUOpcode.SUB,
        OpcodeType.ADD: ALUOpcode.ADD,
        OpcodeType.MOD: ALUOpcode.MOD,
        OpcodeType.EQ: ALUOpcode.EQ,
        OpcodeType.GR: ALUOpcode.GR,
        OpcodeType.LS: ALUOpcode.LS,
    }.get(opcode_type)


class ControlUnit:
    program_memory_size = None
    program_memory = None
    data_path = None
    ps = None
    IO = "h"

    input_tokens: typing.ClassVar[list[tuple]] = []
    tokens_handled: typing.ClassVar[list[bool]] = []

    tick_number = 0
    instruction_number = 0

    def __init__(self, data_path: DataPath, program_memory_size: int, input_tokens: list[tuple]):
        random.seed(17)
        self.data_path = data_path
        self.input_tokens = input_tokens
        self.tokens_handled = [False for _ in input_tokens]
        self.program_memory_size = program_memory_size
        self.program_memory = [{"index": x, "command": 0, "arg": 0} for x in range(self.program_memory_size)]
        self.ps = {"Intr_Req": False, "Intr_On": True}

    def fill_memory(self, opcodes: list) -> None:
        for opcode in opcodes:
            mem_cell = int(opcode["index"])
            assert 0 <= mem_cell < self.program_memory_size, "Program index out of memory size"
            self.program_memory[mem_cell] = opcode

    def signal_latch_pc(self, selector: Selector) -> None:
        if selector is Selector.PC_INC:
            self.data_path.pc += 1
        elif selector is Selector.PC_RET:
            self.data_path.pc = self.data_path.return_stack[self.data_path.i]

    def signal_latch_ps(self, intr_on: bool) -> None:
        self.ps["Intr_On"] = intr_on
        self.ps["Intr_Req"] = self.check_for_interrupts()

    def check_for_interrupts(self) -> bool:
        # example input [(1, 'h'), (10, 'e'), (20, 'l'), (25, 'l'), (100, 'o')]
        if self.ps["Intr_On"]:
            for index, interrupt in enumerate(self.input_tokens):
                if not self.tokens_handled[index] and interrupt[0] >= self.tick:
                    self.IO = interrupt[1]
                    self.ps["Intr_Req"] = True
                    self.tokens_handled[index] = True
        return False

    def tick(self, operations: list[typing.Callable], comment="") -> None:
        self.tick_number += 1
        random.shuffle(operations)
        for operation in operations:
            operation()
        self.__print__(comment)

    def command_cycle(self):
        self.instruction_number += 1
        self.decode_and_execute_instruction()
        self.check_for_interrupts()
        self.signal_latch_pc(Selector.PC_INC)

    def decode_and_execute_instruction(self) -> None:
        memory_cell = self.program_memory[self.data_path.pc]
        command = memory_cell["command"]
        arithmetic_operation = opcode_to_alu_opcode(command)
        if arithmetic_operation is not None:
            self.tick([
                lambda: self.data_path.signal_alu_operation(arithmetic_operation)
            ])
            self.tick([
                lambda: self.data_path.signal_latch_top(Selector.TOP_ALU)
            ])
            self.tick([
                lambda: self.data_path.signal_latch_sp(Selector.SP_DEC)
            ])
            self.tick([
                lambda: self.data_path.signal_latch_next(Selector.NEXT_MEM)
            ])
        elif command == OpcodeType.PUSH:
            self.tick([
                lambda: self.data_path.signal_data_wr()
            ])
            self.tick([
                lambda: self.data_path.signal_latch_sp(Selector.SP_INC),
                lambda: self.data_path.signal_latch_next(Selector.NEXT_TOP)
            ])
            self.tick([
                lambda: self.data_path.signal_latch_top(Selector.TOP_IMMEDIATE, memory_cell["arg"])
            ])

    def __print__(self, comment: str) -> None:
        state_repr = (
            "TICK: {:4} | PS_REQ {:1} | PS_STATE: {:1} | SP: {:3} | I: {:3} | TOP: {:7} | NEXT : {:7} | TEMP: {:7} | "
            "TOP_OF_DS : {:10} | RETURN_STACK[I] {:7} | DATA_MEMORY[TOP] {:7}"
        ).format(
            self.tick_number,
            self.ps["Intr_Req"],
            self.ps["Intr_On"],
            self.data_path.sp,
            self.data_path.i,
            self.data_path.top,
            self.data_path.next,
            self.data_path.temp,
            str(self.data_path.data_stack[self.data_path.sp:self.data_path.sp - 3:-1]),
            self.data_path.return_stack[self.data_path.i],
            self.data_path.memory[self.data_path.top],
        )
        logger.info(state_repr + " " + comment)


def simulation(code: list, limit: int, input_tokens: list[tuple]):
    data_path = DataPath(15000, 15000, 15000)
    control_unit = ControlUnit(data_path, 15000, input_tokens)
    control_unit.fill_memory(code)
    while control_unit.instruction_number < limit:
        control_unit.command_cycle()
        print()
    return ["", control_unit.instruction_number, control_unit.tick_number]


def main(code_path: str, token_path: str | None) -> None:
    input_tokens = []
    if token_path is not None:
        with open(token_path, encoding="utf-8") as file:
            input_text = file.read()
            input_tokens = eval(input_text)
    code = read_code(code_path)
    output, instr_num, ticks = simulation(
        code,
        limit=10,
        input_tokens=input_tokens,
    )
    print(f"Output: {output}\nInstruction number: {instr_num}\nTicks: {ticks - 1}")


if __name__ == "__main__":
    assert 2 <= len(sys.argv) <= 3, "Wrong arguments: machine.py <code_file> [<input_file>]"
    if len(sys.argv) == 3:
        _, code_file, input_file = sys.argv
    else:
        _, code_file = sys.argv
        input_file = None
    main(code_file, input_file)
