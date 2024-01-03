import sys

from isa import Term, TermType, write_code

variables = {}
variable_current_address = 10


def word_to_term(word: str) -> Term | None:
    return {
        'di': TermType.DI,
        'ei': TermType.EI,
        'dup': TermType.DUP,
        '+': TermType.ADD,
        '-': TermType.SUB,
        '*': TermType.MUL,
        '/': TermType.DIV,
        'mod': TermType.MOD,
        'omit': TermType.OMIT,
        'swap': TermType.SWAP,
        'drop': TermType.DROP,
        'over': TermType.OVER,
        '=': TermType.EQ,
        '<': TermType.LS,
        '>': TermType.GR,
        'variable': TermType.VARIABLE,
        'allot': TermType.ALLOT,
        '!': TermType.STORE,
        '@': TermType.LOAD,
        'if': TermType.IF,
        'else': TermType.ELSE,
        'then': TermType.THEN,
        '.': TermType.PRINT,
        ':': TermType.DEF,
        ';': TermType.RET,
        ':intr': TermType.DEF_INTR,
        'do': TermType.DO,
        'loop': TermType.LOOP,
        'begin': TermType.BEGIN,
        'until': TermType.UNTIL,
        'i': TermType.LOOP_CNT
    }.get(word)


def split_to_terms(source_code: str) -> list[Term]:
    code_words = source_code.split(' ')
    terms = []
    for word_number, word in enumerate(code_words):
        term_type = word_to_term(word)
        terms.append(Term(word_number, term_type, word))
    return terms


def set_closed_indexes(terms: list[Term], begin: TermType, end: TermType, error_message: str) -> None:
    nested = []
    for term_index, term in enumerate(terms):
        if term.term_type is begin:
            nested.append(term_index)
        if term.term_type == end:
            term.opened = nested.pop()
        assert len(nested) >= 0, error_message
    assert len(nested) == 0, error_message


def set_closed_indexes_func(terms: list[Term]) -> None:
    func_indexes = []
    error_message = "Unbalanced : and ; or :intr and ;"
    for term_index, term in enumerate(terms):
        if term.term_type is TermType.DEF or term.term_type is TermType.DEF_INTR:
            func_indexes.append(term_index)
        if term.term_type == TermType.RET:
            term.opened = func_indexes.pop()
        assert 0 <= len(func_indexes) < 1, error_message
    assert len(func_indexes) == 0, error_message


def assert_with_line(cond: bool, msg: str, word_index: int) -> None:
    assert cond, msg + '. word: ' + str(word_index)


def set_variables(terms: list[Term]) -> None:
    global variable_current_address
    for term_index, term in enumerate(terms):

        # variable <name> [<size> allot]
        if term.term_type is TermType.VARIABLE:
            assert_with_line(term_index + 1 < len(terms), 'Bad variable declaration', term.word_number)
            assert_with_line(terms[term_index + 1].term_type is None, 'Variable name same as key', term.word_number + 1)
            assert_with_line(terms[term_index + 1].word[0].isalpha(), 'Bad variable name', term.word_number + 1)
            assert_with_line(terms[term_index + 1] not in variables, 'Variable already exists', term.word_number + 1)
            allot_size = 1
            if term_index + 3 < len(terms) and terms[term_index + 3].term_type is TermType.ALLOT:
                terms[term_index + 2].converted = True
                try:
                    allot_size = int(terms[term_index + 2].word)
                    assert_with_line(1 <= allot_size <= 100, 'Incorrect allot size', term.word_number + 2)
                    variables[terms[term_index + 1].word] = variable_current_address
                    variable_current_address += allot_size
                except ValueError:
                    assert_with_line(True, 'Incorrect allot size', term.word_number + 2)
            variables[terms[term_index + 1].word] = variable_current_address
            variable_current_address += allot_size
            terms[term_index + 1].converted = True

    # replace variable name by actual address
    for term_index, term in enumerate(terms):
        if term.term_type is None and not term.converted:
            if term.word in variables:
                term.word = str(variables[term.word])


def set_if_else_then(terms: list[Term]) -> None:
    nested_ifs = []
    for term_index, term in enumerate(terms):
        if term.term_type is TermType.IF:
            nested_ifs.append(term)
        elif term.term_type is TermType.ELSE:
            nested_ifs.append(term)
        elif term.term_type is TermType.THEN:
            last_if = nested_ifs.pop()
            assert_with_line(last_if is not None, 'IF-ELSE-THEN unbalanced', term.word_number)
            if last_if.term_type is TermType.ELSE:
                last_else = last_if
                last_if = nested_ifs.pop()
                assert_with_line(last_if is not None, 'IF-ELSE-THEN unbalanced', term.word_number)
                last_else.operand = term
                last_if.operand = last_else
            else:
                last_if.operand = term

    assert_with_line(len(nested_ifs) == 0, 'IF-ELSE-THEN unbalanced', nested_ifs[0].word_number)


def validate_and_fix_terms(terms: list[Term]) -> None:
    set_closed_indexes(terms, TermType.DO, TermType.LOOP, 'Unbalanced do ... loop')
    set_closed_indexes(terms, TermType.BEGIN, TermType.UNTIL, 'Unbalanced begin ... until')
    set_closed_indexes_func(terms)
    set_variables(terms)
    set_if_else_then(terms)


def translate(source_code: str) -> str:
    terms = split_to_terms(source_code)
    validate_and_fix_terms(terms)
    return 'lol...'


def main(source_file: str, target_file: str) -> None:
    with open(source_file, encoding='utf-8') as f:
        source_code = f.read()
    code = translate(source_code)
    write_code(target_file, code)
    print('source LoC:', len(source_code.split('\n')), 'code instr:', len(code))


if __name__ == '__main__':
    assert len(sys.argv) == 3, 'Wrong arguments: translator.py <input_file> <target_file>'
    _, source, target = sys.argv
    main(source, target)
