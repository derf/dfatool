from sly import Lexer, Parser


class TimedWordLexer(Lexer):
    tokens = {LPAREN, RPAREN, IDENTIFIER, NUMBER, ARGSEP, FUNCTIONSEP}
    ignore = ' \t'

    LPAREN = r'\('
    RPAREN = r'\)'
    IDENTIFIER = r'[a-zA-Z_][a-zA-Z0-9_]*'
    NUMBER = r'[0-9e.]+'
    ARGSEP = r','
    FUNCTIONSEP = r';'


class TimedWordParser(Parser):
    tokens = TimedWordLexer.tokens

    @_('timedSymbol FUNCTIONSEP timedWord')
    def timedWord(self, p):
        ret = [p.timedSymbol]
        ret.extend(p.timedWord)
        return ret

    @_('timedSymbol FUNCTIONSEP', 'timedSymbol')
    def timedWord(self, p):
        return [p.timedSymbol]

    @_('IDENTIFIER', 'IDENTIFIER LPAREN RPAREN')
    def timedSymbol(self, p):
        return (p.IDENTIFIER,)

    @_('IDENTIFIER LPAREN args RPAREN')
    def timedSymbol(self, p):
        return (p.IDENTIFIER, *p.args)

    @_('arg ARGSEP args')
    def args(self, p):
        ret = [p.arg]
        ret.extend(p.args)
        return ret

    @_('arg')
    def args(self, p):
        return [p.arg]

    @_('NUMBER')
    def arg(self, p):
        return [float(p.NUMBER)]

    @_('IDENTIFIER')
    def arg(self, p):
        return [p.IDENTIFIER]


if __name__ == '__main__':
    data = 'init(); sleep(12345); foo(_, 7);'
    lexer = TimedWordLexer()
    parser = TimedWordParser()
    for tok in lexer.tokenize(data):
        print('type={}, value={}'.format(tok.type, tok.value))
    print(parser.parse(lexer.tokenize(data)))
