from enum import Enum


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
    alu_operations = [
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

    def calc(self) -> None:
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
            assert False, f"Unknown ALU operation: {self.operation}"

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

    input_buffer = None
    output_buffer = None

    alu = None

    def __init__(self, memory_size: int, data_stack_size: int, return_stack_size: int, input_buffer: list):
        assert memory_size > 0, "Data memory size must be greater than zero"
        assert data_stack_size > 0, "Data stack size must be greater than zero"
        assert return_stack_size > 0, "Return stack size must be greater than zero"

        self.memory_size = memory_size
        self.data_stack_size = data_stack_size
        self.return_stack_size = return_stack_size
        self.memory = [0] * memory_size
        self.data_stack = [0] * data_stack_size
        self.return_stack = [0] * return_stack_size

        self.sp = 0
        self.i = 0
        self.pc = 0
        self.top = self.next = self.temp = 0
        self.input_buffer = input_buffer
        self.output_buffer = []

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

    def signal_latch_top(self, selector: Selector) -> None:
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


class ControlUnit:
    program_memory_size = None
    program_memory = None
    data_path = None
    ps = None

    def __init__(self, data_path: DataPath, program_memory_size: int):
        self.data_path = data_path
        self.program_memory_size = program_memory_size
        self.program_memory = [{"index": x, "value": 0} for x in range(self.program_memory_size)]
        self.ps = {"Intr_Req": False, "Intr_On": True}

    def signal_fill_memory(self, opcodes: list) -> None:
        for opcode in opcodes:
            mem_cell = opcode["index"]
            assert 0 <= mem_cell < self.program_memory_size, "Program index out of memory size"
            self.program_memory[self.program_memory_size] = opcode

    def signal_latch_pc(self, selector: Selector) -> None:
        if selector is Selector.PC_INC:
            self.data_path.pc += 1
        elif selector is Selector.PC_RET:
            self.data_path.pc = self.data_path.return_stack[self.data_path.i]

    def signal_latch_ps(self, intr_on: bool) -> None:
        self.ps["Intr_On"] = intr_on
        self.ps["Intr_Req"] = self.check_for_interrupts()

    def check_for_interrupts(self) -> bool:
        pass
