from __future__ import annotations

import shlex
import sys

from isa import Opcode, OpcodeType, Term, TermType, write_code

variables = {}
variable_current_address = 10
functions = {}
intr_function = None


def word_to_term(word: str) -> Term | None:
    return {
        "di": TermType.DI,
        "ei": TermType.EI,
        "dup": TermType.DUP,
        "+": TermType.ADD,
        "-": TermType.SUB,
        "*": TermType.MUL,
        "/": TermType.DIV,
        "mod": TermType.MOD,
        "omit": TermType.OMIT,
        "swap": TermType.SWAP,
        "drop": TermType.DROP,
        "over": TermType.OVER,
        "=": TermType.EQ,
        "<": TermType.LS,
        ">": TermType.GR,
        "variable": TermType.VARIABLE,
        "allot": TermType.ALLOT,
        "!": TermType.STORE,
        "@": TermType.LOAD,
        "if": TermType.IF,
        "else": TermType.ELSE,
        "then": TermType.THEN,
        ".": TermType.PRINT,
        ":": TermType.DEF,
        ";": TermType.RET,
        ":intr": TermType.DEF_INTR,
        "do": TermType.DO,
        "loop": TermType.LOOP,
        "begin": TermType.BEGIN,
        "until": TermType.UNTIL,
        "i": TermType.LOOP_CNT,
    }.get(word)


def split_to_terms(source_code: str) -> list[Term]:
    code_words = shlex.split(source_code.replace("\n", " "), posix=True)
    code_words = list(filter(lambda x: len(x) > 0, code_words))
    terms = []
    for word_number, word in enumerate(code_words):
        term_type = word_to_term(word)
        if word[:2] == ". ":
            word = f'."{word[2:]}"'
            term_type = TermType.STRING
        terms.append(Term(word_number, term_type, word))
    return terms


def set_closed_indexes(terms: list[Term], begin: TermType, end: TermType, error_message: str) -> None:
    nested = []
    for term_index, term in enumerate(terms):
        if term.term_type is begin:
            nested.append(term_index)
        if term.term_type == end:
            assert len(nested) > 0, error_message + " at word #" + str(term.word_number)
            term.opened = nested.pop()
    assert len(nested) == 0, error_message


def set_functions(terms: list[Term]) -> None:
    global functions, intr_function
    func_indexes = []
    for term_index, term in enumerate(terms):
        if term.term_type is TermType.DEF or term.term_type is TermType.DEF_INTR:
            assert term_index + 1 < len(terms), "Missed function name" + str(term.word_number)
            assert len(func_indexes) == 0, "Unclosed function at word #" + str(term.word_number)
            assert term.word not in functions, "Duplicate function at word #" + str(term.word_number)
            func_indexes.append(term_index)
            func_name = terms[term_index + 1].word
            functions[func_name] = term.word_number + 2
            terms[term_index + 1].converted = True
            if term.term_type is TermType.DEF_INTR:
                intr_function = func_name
        if term.term_type == TermType.RET:
            assert len(func_indexes) >= 1, "RET out of function at word #" + str(term.word_number)
            term.opened = func_indexes.pop()
    assert len(func_indexes) == 0, "Unclosed function"


def assert_with_line(cond: bool, msg: str, word_index: int) -> None:
    assert cond, msg + ". word: " + str(word_index)


def set_variables(terms: list[Term]) -> None:
    global variable_current_address
    for term_index, term in enumerate(terms):
        # variable <name> [<size> allot]
        if term.term_type is TermType.VARIABLE:
            assert term_index + 1 < len(terms), "Bad variable declaration at word #" + str(term.word_number)
            assert terms[term_index + 1].term_type is None, "Variable name same as key at word #" + str(
                term.word_number + 1
            )
            assert terms[term_index + 1].word[0].isalpha(), "Bad variable name at word #" + str(term.word_number + 1)
            assert terms[term_index + 1] not in variables, "Variable already exists at word #" + str(
                term.word_number + 1
            )
            variables[terms[term_index + 1].word] = variable_current_address
            variable_current_address += 1
            terms[term_index + 1].converted = True
            if term_index + 3 < len(terms) and terms[term_index + 3].term_type is TermType.ALLOT:
                set_allot_for_variable(terms, term_index + 3)


def set_allot_for_variable(terms: list[Term], term_index: int) -> None:
    global variable_current_address
    assert term_index + 3 < len(terms), "Bad allot declaration"
    term = terms[term_index]
    if term.term_type is TermType.ALLOT:
        assert term_index - 3 >= 0, "Bad allot declaration at word #" + str(term.word_number)
        terms[term_index - 1].converted = True
        try:
            allot_size = int(terms[term_index - 1].word)
            assert 1 <= allot_size <= 100, "Incorrect allot size at word #" + str(term.word_number - 1)
            variable_current_address += allot_size
        except ValueError:
            assert True, "Incorrect allot size at word #" + str(term.word_number - 1)


def set_if_else_then(terms: list[Term]) -> None:
    nested_ifs = []
    for term_index, term in enumerate(terms):
        if term.term_type is TermType.IF:
            nested_ifs.append(term)
        elif term.term_type is TermType.ELSE:
            nested_ifs.append(term)
        elif term.term_type is TermType.THEN:
            assert len(nested_ifs) > 0, "IF-ELSE-THEN unbalanced at word #" + str(term.word_number)
            last_if = nested_ifs.pop()
            if last_if.term_type is TermType.ELSE:
                last_else = last_if
                assert len(nested_ifs) > 0, "IF-ELSE-THEN unbalanced at word #" + str(term.word_number)
                last_if = nested_ifs.pop()
                last_else.operand = term.word_number + 1
                last_if.operand = last_else.word_number + 1
            else:
                last_if.operand = term.word_number + 1

    assert len(nested_ifs) == 0, "IF-ELSE-THEN unbalanced at word #" + str(nested_ifs[0].word_number)


def replace_vars_funcs(terms: list[Term]) -> None:
    for term_index, term in enumerate(terms):
        if term.term_type is None and not term.converted:
            if term.word in variables:
                term.word = str(variables[term.word])
    for term_index, term in enumerate(terms):
        if term.term_type is None and not term.converted:
            if term.word in functions.keys():
                term.operand = functions[term.word]
                term.term_type = TermType.CALL
                term.word = "call"


def validate_and_fix_terms(terms: list[Term]):
    set_closed_indexes(terms, TermType.DO, TermType.LOOP, "Unbalanced do ... loop")
    set_closed_indexes(terms, TermType.BEGIN, TermType.UNTIL, "Unbalanced begin ... until")
    set_functions(terms)
    set_variables(terms)
    replace_vars_funcs(terms)
    set_if_else_then(terms)


def term_to_opcode(term: Term) -> Opcode:
    opcode = {
        TermType.DI: [Opcode(OpcodeType.DI, [])],
        TermType.EI: [Opcode(OpcodeType.EI, [])],
        TermType.DUP: [Opcode(OpcodeType.DUP, [])],
        TermType.ADD: [Opcode(OpcodeType.ADD, [])],
        TermType.SUB: [Opcode(OpcodeType.SUB, [])],
        TermType.MUL: [Opcode(OpcodeType.MUL, [])],
        TermType.DIV: [Opcode(OpcodeType.DIV, [])],
        TermType.MOD: [Opcode(OpcodeType.MOD, [])],
        TermType.OMIT: [Opcode(OpcodeType.OMIT, [])],
        TermType.SWAP: [Opcode(OpcodeType.SWAP, [])],
        TermType.DROP: [Opcode(OpcodeType.DROP, [])],
        TermType.OVER: [Opcode(OpcodeType.OVER, [])],
        TermType.EQ: [Opcode(OpcodeType.EQ, [])],
        TermType.LS: [Opcode(OpcodeType.LS, [])],
        TermType.GR: [Opcode(OpcodeType.GR, [])],
        TermType.READ: [Opcode(OpcodeType.READ, [])],
        TermType.VARIABLE: [],
        TermType.ALLOT: [],
        TermType.STORE: [Opcode(OpcodeType.STORE, [])],
        TermType.LOAD: [Opcode(OpcodeType.LOAD, [])],
        TermType.IF: [Opcode(OpcodeType.ZJMP, [])],
        TermType.ELSE: [Opcode(OpcodeType.JMP, [])],
        TermType.THEN: [],
        TermType.PRINT: [Opcode(OpcodeType.WRITE, [])],
        TermType.DEF: [],
        TermType.RET: [Opcode(OpcodeType.RET, [])],
        TermType.DEF_INTR: [],
        TermType.DO: [
            Opcode(OpcodeType.POP, []),  # R(i)
            Opcode(OpcodeType.POP, []),  # R(i, n)
        ],
        TermType.LOOP: [
            Opcode(OpcodeType.RPOP, []),  # (n)
            Opcode(OpcodeType.RPOP, []),  # (n, i)
            Opcode(OpcodeType.PUSH, [1]),  # (n, i, 1)
            Opcode(OpcodeType.ADD, []),  # (n, i + 1)
            Opcode(OpcodeType.OVER, []),  # (n, i + 1, n)
            Opcode(OpcodeType.OVER, []),  # (n, i + 1, n, i + 1)
            Opcode(OpcodeType.GR, []),  # (n, i + 1, n > i + 1 [i + 1 < n])
            Opcode(OpcodeType.ZJMP, []),  # (n, i + 1)
            Opcode(OpcodeType.DROP, []),  # (n)
            Opcode(OpcodeType.DROP, []),  # ()
        ],
        TermType.BEGIN: [],
        TermType.UNTIL: [Opcode(OpcodeType.ZJMP, [])],
        TermType.LOOP_CNT: [
            Opcode(OpcodeType.RPOP, []),
            Opcode(OpcodeType.DUP, []),
            Opcode(OpcodeType.RPOP, []),
        ],
        TermType.CALL: [Opcode(OpcodeType.CALL, [])],
    }.get(term.term_type)

    if term.operand:
        opcode.params.append(term.operand)

    return opcode


def terms_to_opcodes(terms: list[Term]) -> list[Opcode]:
    return list(map(term_to_opcode, terms))


def translate(source_code: str) -> str:
    terms = split_to_terms(source_code)
    validate_and_fix_terms(terms)
    opcodes = terms_to_opcodes(terms)
    for i in list(map(lambda x: str(x.word_number) + ": " + x.word + " --> " + str(x.operand), terms)):
        print(i)
    print(variables, functions)
    return "lol..."


def main(source_file: str, target_file: str) -> None:
    with open(source_file, encoding="utf-8") as f:
        source_code = f.read()
    code = translate(source_code)
    write_code(target_file, code)
    print("source LoC:", len(source_code.split("\n")), "code instr:", len(code))


if __name__ == "__main__":
    assert len(sys.argv) == 3, "Wrong arguments: translator.py <input_file> <target_file>"
    _, source, target = sys.argv
    main(source, target)
