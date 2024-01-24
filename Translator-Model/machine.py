from __future__ import annotations

import logging
import sys
import typing
from enum import Enum

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
    PC_IMMEDIATE = "pc_immediate"

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
            self.result = self.src_b // self.src_a
        elif self.operation == ALUOpcode.SUB:
            self.result = self.src_b - self.src_a
        elif self.operation == ALUOpcode.MOD:
            self.result = self.src_b % self.src_a
        elif self.operation == ALUOpcode.EQ:
            self.result = int(self.src_a == self.src_b)
        elif self.operation == ALUOpcode.GR:
            self.result = int(self.src_a < self.src_b)
        elif self.operation == ALUOpcode.LS:
            self.result = int(self.src_a >= self.src_b)

    def set_details(self, src_a, src_b, operation: ALUOpcode) -> None:
        self.src_a = src_a
        self.src_b = src_b
        self.operation = operation


class DataPath:
    def __init__(self, memory_size: int, data_stack_size: int, return_stack_size: int):
        assert memory_size > 0, "Data memory size must be greater than zero"
        assert data_stack_size > 0, "Data stack size must be greater than zero"
        assert return_stack_size > 0, "Return stack size must be greater than zero"

        self.memory_size = memory_size
        self.data_stack_size = data_stack_size
        self.return_stack_size = return_stack_size
        self.memory = [4747] * memory_size
        self.data_stack = [8877] * data_stack_size
        self.return_stack = [9988] * return_stack_size

        self.sp = 4
        self.i = 4
        self.pc = 0
        self.top = self.next = self.temp = 8877
        self.sr = [0 for _ in range(32)]
        self.sro = [0 for _ in range(32)]
        self.sri = [0 for _ in range(32)]

        self.latch_sr = False

        self.alu = ALU()

    def signal_latch_sr(self, signal: bool) -> None:
        self.latch_sr = signal
        for i in range(len(self.sro)):
            self.sro[i] = int(self.next / pow(2, i)) % 2

    def signal_latch_sri(self) -> None:
        self.sri = self.sr.copy()

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

    def signal_mem_write(self) -> None:
        assert self.top >= 0, "Address below 0"
        assert self.top < self.memory_size, "Memory overflow"
        self.memory[self.top] = self.next

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
    def __init__(self, data_path: DataPath, program_memory_size: int, input_tokens: list[tuple]):
        self.data_path = data_path
        self.input_tokens = input_tokens
        self.tokens_handled = [False for _ in input_tokens]
        self.program_memory_size = program_memory_size
        self.program_memory = [{"index": x, "command": "nop", "arg": 0} for x in range(self.program_memory_size)]
        self.ps = {"Intr_Req": False, "Intr_On": True, "Intr_Mode": False}
        self.IO = ""
        self.tick_number = 0
        self.instruction_number = 0
        self.out_buffer = []
        self.cs = True

    def fill_memory(self, opcodes: list) -> None:
        for opcode in opcodes:
            mem_cell = int(opcode["index"])
            assert 0 <= mem_cell < self.program_memory_size, "Program index out of memory size"
            self.program_memory[mem_cell] = opcode

    def signal_latch_pc(self, selector: Selector, immediate=0) -> None:
        if selector is Selector.PC_INC:
            self.data_path.pc += 1
        elif selector is Selector.PC_RET:
            self.data_path.pc = self.data_path.return_stack[self.data_path.i]
        elif selector is Selector.PC_IMMEDIATE:
            self.data_path.pc = immediate - 1

    def signal_latch_ps(self, intr_on=None, intr_mode=None):
        if intr_on is not None:
            self.ps["Intr_On"] = intr_on
        if intr_mode is not None:
            self.ps["Intr_Mode"] = intr_mode

    def check_for_interrupts(self) -> None:
        if self.ps["Intr_Req"] and self.ps["Intr_On"]:
            self.ps["Intr_On"] = False
            self.ps["Intr_Req"] = False
            self.tick([lambda: self.data_path.signal_ret_wr(Selector.RET_STACK_PC)])
            self.tick(
                [
                    lambda: self.signal_latch_pc(Selector.PC_IMMEDIATE, 1),
                    lambda: self.data_path.signal_latch_i(Selector.I_INC),
                ]
            )
            self.ps["Intr_Mode"] = True

    def get_io_value(self) -> int:
        io_value = 0
        for i, bit in enumerate(self.data_path.sri):
            io_value += pow(2, i) * bit
        return io_value

    def tick(self, operations: list[typing.Callable]) -> None:
        self.tick_number += 1
        for operation in operations:
            operation()
        self.__print__(self.program_memory[self.data_path.pc]["command"])

    def command_cycle(self):
        self.instruction_number += 1
        bubble = self.decode_and_execute_instruction()
        if not bubble:
            self.check_for_interrupts()
            self.signal_latch_pc(Selector.PC_INC)

    def get_instruction_ticks(self, memory_cell) -> list[list[typing.Callable]]:
        command = memory_cell["command"]
        arithmetic_operation = opcode_to_alu_opcode(OpcodeType(command))
        alu_operations = [
            [lambda: self.data_path.signal_alu_operation(arithmetic_operation)],
            [lambda: self.data_path.signal_latch_top(Selector.TOP_ALU)],
            [lambda: self.data_path.signal_latch_sp(Selector.SP_DEC)],
            [lambda: self.data_path.signal_latch_next(Selector.NEXT_MEM)],
        ]
        return {
            OpcodeType.ADD: alu_operations,
            OpcodeType.MOD: alu_operations,
            OpcodeType.SUB: alu_operations,
            OpcodeType.MUL: alu_operations,
            OpcodeType.DIV: alu_operations,
            OpcodeType.GR: alu_operations,
            OpcodeType.EQ: alu_operations,
            OpcodeType.LS: alu_operations,
            OpcodeType.PUSH: [
                [lambda: self.data_path.signal_data_wr()],
                [
                    lambda: self.data_path.signal_latch_sp(Selector.SP_INC),
                    lambda: self.data_path.signal_latch_next(Selector.NEXT_TOP),
                ],
                [lambda: self.data_path.signal_latch_top(Selector.TOP_IMMEDIATE, memory_cell["arg"])],
            ],
            OpcodeType.DROP: [
                [
                    lambda: self.data_path.signal_latch_top(Selector.TOP_NEXT),
                    lambda: self.data_path.signal_latch_sp(Selector.SP_DEC),
                ],
                [lambda: self.data_path.signal_latch_next(Selector.NEXT_MEM)],
            ],
            OpcodeType.OMIT: [
                [
                    lambda: self.data_path.signal_latch_sr(True),
                    lambda: self.data_path.signal_latch_top(Selector.TOP_NEXT),
                    lambda: self.data_path.signal_latch_sp(Selector.SP_DEC),
                ],
                [lambda: self.data_path.signal_latch_next(Selector.NEXT_MEM)],
                [
                    lambda: self.data_path.signal_latch_top(Selector.TOP_NEXT),
                    lambda: self.data_path.signal_latch_sp(Selector.SP_DEC),
                ],
                [lambda: self.data_path.signal_latch_next(Selector.NEXT_MEM)],
            ],
            OpcodeType.READ: [[lambda: self.data_path.signal_latch_top(Selector.TOP_IMMEDIATE, self.get_io_value())]],
            OpcodeType.SWAP: [
                [lambda: self.data_path.signal_latch_temp(Selector.TEMP_TOP)],
                [lambda: self.data_path.signal_latch_top(Selector.TOP_NEXT)],
                [lambda: self.data_path.signal_latch_next(Selector.NEXT_TEMP)],
            ],
            OpcodeType.OVER: [
                [lambda: self.data_path.signal_data_wr()],
                [
                    lambda: self.data_path.signal_latch_temp(Selector.TEMP_TOP),
                    lambda: self.data_path.signal_latch_sp(Selector.SP_INC),
                ],
                [lambda: self.data_path.signal_latch_top(Selector.TOP_NEXT)],
                [lambda: self.data_path.signal_latch_next(Selector.NEXT_TEMP)],
            ],
            OpcodeType.DUP: [
                [lambda: self.data_path.signal_data_wr()],
                [
                    lambda: self.data_path.signal_latch_next(Selector.NEXT_TOP),
                    lambda: self.data_path.signal_latch_sp(Selector.SP_INC),
                ],
            ],
            OpcodeType.LOAD: [[lambda: self.data_path.signal_latch_top(Selector.TOP_MEM)]],
            OpcodeType.STORE: [
                [lambda: self.data_path.signal_mem_write(), lambda: self.data_path.signal_latch_sp(Selector.SP_DEC)],
                [lambda: self.data_path.signal_latch_next(Selector.NEXT_MEM)],
                [
                    lambda: self.data_path.signal_latch_top(Selector.TOP_NEXT),
                    lambda: self.data_path.signal_latch_sp(Selector.SP_DEC),
                ],
                [lambda: self.data_path.signal_latch_next(Selector.NEXT_MEM)],
            ],
            OpcodeType.POP: [
                [lambda: self.data_path.signal_latch_temp(Selector.TEMP_TOP)],
                [
                    lambda: self.data_path.signal_latch_top(Selector.TOP_NEXT),
                    lambda: self.data_path.signal_latch_sp(Selector.SP_DEC),
                ],
                [
                    lambda: self.data_path.signal_latch_next(Selector.NEXT_MEM),
                    lambda: self.data_path.signal_ret_wr(Selector.RET_STACK_OUT),
                ],
                [lambda: self.data_path.signal_latch_i(Selector.I_INC)],
            ],
            OpcodeType.RPOP: [
                [lambda: self.data_path.signal_latch_i(Selector.I_DEC)],
                [
                    lambda: self.data_path.signal_latch_temp(Selector.TEMP_RETURN),
                    lambda: self.data_path.signal_data_wr(),
                ],
                [
                    lambda: self.data_path.signal_latch_next(Selector.NEXT_TOP),
                    lambda: self.data_path.signal_latch_sp(Selector.SP_INC),
                ],
                [lambda: self.data_path.signal_latch_top(Selector.TOP_TEMP)],
            ],
            OpcodeType.ZJMP: [
                [
                    lambda: self.data_path.signal_latch_top(Selector.TOP_NEXT),
                    lambda: self.data_path.signal_latch_sp(Selector.SP_DEC),
                ],
                [lambda: self.data_path.signal_latch_next(Selector.NEXT_MEM)],
            ],
            OpcodeType.JMP: [[lambda: self.signal_latch_pc(Selector.PC_IMMEDIATE, memory_cell["arg"])]],
            OpcodeType.CALL: [
                [lambda: self.data_path.signal_ret_wr(Selector.RET_STACK_PC)],
                [
                    lambda: self.data_path.signal_latch_i(Selector.I_INC),
                    lambda: self.signal_latch_pc(Selector.PC_IMMEDIATE, memory_cell["arg"]),
                ],
            ],
            OpcodeType.DI: [[lambda: self.signal_latch_ps(intr_on=False)]],
            OpcodeType.EI: [[lambda: self.signal_latch_ps(intr_on=True)]],
            OpcodeType.RET: [
                [lambda: self.data_path.signal_latch_i(Selector.I_DEC)],
                [lambda: self.signal_latch_pc(Selector.PC_RET), lambda: self.signal_latch_ps(intr_mode=False)],
            ],
        }.get(command)

    def decode_and_execute_instruction(self) -> bool:
        memory_cell = self.program_memory[self.data_path.pc]
        command = memory_cell["command"]

        # wait until previous output was completed
        if command == OpcodeType.OMIT and not self.cs:
            self.tick_number += 1
            return True

        # add additional action to zjmp if top is 0
        if command == OpcodeType.ZJMP and self.data_path.top == 0:
            self.tick(
                [
                    lambda: self.signal_latch_pc(Selector.PC_IMMEDIATE, memory_cell["arg"]),
                ]
            )

        if command == OpcodeType.HALT:
            raise StopIteration

        instruction_ticks = self.get_instruction_ticks(memory_cell)
        for operations in instruction_ticks:
            self.tick(operations)

        return False

    def __print__(self, command: str) -> None:
        tos_memory = self.data_path.data_stack[self.data_path.sp - 1 : self.data_path.sp - 4 : -1]
        tos = [self.data_path.top, self.data_path.next, *tos_memory]
        ret_tos = self.data_path.return_stack[self.data_path.i - 1 : self.data_path.i - 4 : -1]
        state_repr = (
            "{:4} | TICK: {:4} | INSTR: {:7} | PC: {:3} | PS_REQ {:1} | PS_STATE: {:1} | SP: {:3} | I: "
            "{:3} | TEMP: {:7} | DATA_MEMORY[TOP] {:7} | TOS : {} | RETURN_TOS : {}"
        ).format(
            "INTR" if self.ps["Intr_Mode"] else "MAIN",
            self.tick_number,
            self.data_path.pc,
            command,
            self.ps["Intr_Req"],
            self.ps["Intr_On"],
            self.data_path.sp,
            self.data_path.i,
            self.data_path.temp,
            self.data_path.memory[self.data_path.top] if self.data_path.top < self.data_path.memory_size else "?",
            str(tos),
            str(ret_tos),
        )
        logger.info(state_repr)


class MasterSPI:
    def __init__(self, control_unit: ControlUnit, limit: int):
        self.shift_register = [0 for _ in range(32)]
        self.control_unit = control_unit
        self.tick_limit = limit
        self.tick_number = 0
        self.sig_read = False
        self.MISO = False
        self.MOSI = False

    def tick_rise(self) -> None:
        sr = self.control_unit.data_path.sr
        self.MOSI = self.shift_register[len(self.shift_register) - 1]
        self.MISO = sr[len(sr) - 1]

    def tick_fall(self) -> None:
        latch_sr = self.control_unit.data_path.latch_sr
        if latch_sr and not self.sig_read:  # slave
            self.control_unit.data_path.latch_sr = False
            self.control_unit.data_path.sr = self.control_unit.data_path.sro.copy()
            self.tick_number = 0
            self.control_unit.cs = False
            self.sig_read = True
        elif not self.control_unit.cs:  # master + slave
            for i in reversed(range(1, len(self.shift_register))):
                self.shift_register[i] = self.shift_register[i - 1]
            for i in reversed(range(1, len(self.control_unit.data_path.sr))):
                self.control_unit.data_path.sr[i] = self.control_unit.data_path.sr[i - 1]
            self.shift_register[0] = self.MISO
            self.control_unit.data_path.sr[0] = self.MOSI
        if not self.control_unit.cs and self.tick_number >= 32:
            self.control_unit.cs = True
            self.write_content()
            self.tick_number = 0
            if not self.sig_read:
                self.control_unit.data_path.signal_latch_sri()
                self.control_unit.ps["Intr_Req"] = True
            self.sig_read = False

    def check_for_input(self) -> None:
        if self.control_unit.ps["Intr_On"] and self.control_unit.cs and not self.control_unit.ps["Intr_Mode"]:
            for index, interrupt in enumerate(self.control_unit.input_tokens):
                if not self.control_unit.tokens_handled[index] and interrupt[0] <= self.control_unit.tick_number:
                    input_content = ord(interrupt[1])
                    logger.debug("inputted: %s", input_content)
                    for i in range(len(self.shift_register)):
                        self.shift_register[i] = int(input_content / pow(2, i)) % 2
                    self.control_unit.tokens_handled[index] = True
                    self.control_unit.cs = False
                    self.tick_number = 1
                    return

    def tick(self):
        self.tick_rise()
        self.tick_fall()
        self.tick_number += 1
        self.__print__()
        self.check_for_input()

    def __print__(self):
        state_repr = (
            "SPI  | TICK: {:4} | TICK_SPI: {:4} | PS_REQ {:1} | PS_STATE: {:1} | MISO: {:1} | MOSI: {:1} "
            "| CS: {:1} | MSR: {:32} SSR: {:32}"
        ).format(
            self.control_unit.tick_number,
            self.tick_number,
            self.control_unit.ps["Intr_Req"],
            self.control_unit.ps["Intr_On"],
            self.MISO,
            self.MOSI,
            self.control_unit.cs,
            "".join(list(map(str, self.shift_register))),
            "".join(list(map(str, self.control_unit.data_path.sr))),
        )
        logger.info(state_repr)

    def write_content(self):
        if self.sig_read:
            io_value = 0
            for i, bit in enumerate(self.shift_register):
                io_value += pow(2, i) * bit
            self.control_unit.out_buffer.append(io_value)


def simulation(code: list, limit: int, input_tokens: list[tuple]):
    data_path = DataPath(15000, 15000, 15000)
    control_unit = ControlUnit(data_path, 15000, input_tokens)
    control_unit.fill_memory(code)
    master_spi = MasterSPI(control_unit, 10000)
    while control_unit.tick_number < limit:
        try:
            control_unit.command_cycle()
            master_spi.tick()
        except StopIteration:
            break

    # wait for end of user input
    for i in range(32):
        master_spi.tick()

    return [control_unit.out_buffer, control_unit.instruction_number, control_unit.tick_number]


def main(code_path: str, token_path: str | None) -> None:
    input_tokens = []
    if token_path is not None:
        with open(token_path, encoding="utf-8") as file:
            input_text = file.read()
            input_tokens = eval(input_text)
    code = read_code(code_path)
    output, instr_num, ticks = simulation(
        code,
        limit=5000,
        input_tokens=input_tokens,
    )
    output = list(map(chr, output))
    print(f"Output: {''.join(output)}\nInstruction number: {instr_num!s}\nTicks: {ticks - 1!s}")


if __name__ == "__main__":
    assert 2 <= len(sys.argv) <= 3, "Wrong arguments: machine.py <code_file> [<input_file>]"
    if len(sys.argv) == 3:
        _, code_file, input_file = sys.argv
    else:
        _, code_file = sys.argv
        input_file = None
    main(code_file, input_file)
