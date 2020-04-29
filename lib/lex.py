from .sly import Lexer, Parser


class TimedWordLexer(Lexer):
    tokens = {LPAREN, RPAREN, IDENTIFIER, NUMBER, ARGSEP, FUNCTIONSEP}
    ignore = ' \t'

    LPAREN = r'\('
    RPAREN = r'\)'
    IDENTIFIER = r'[a-zA-Z_][a-zA-Z0-9_]*'
    NUMBER = r'[0-9e.]+'
    ARGSEP = r','
    FUNCTIONSEP = r';'


class TimedSequenceLexer(Lexer):
    tokens = {LPAREN, RPAREN, LBRACE, RBRACE, CYCLE, IDENTIFIER, NUMBER, ARGSEP, FUNCTIONSEP}
    ignore = ' \t'

    LPAREN = r'\('
    RPAREN = r'\)'
    LBRACE = r'\{'
    RBRACE = r'\}'
    CYCLE = r'cycle'
    IDENTIFIER = r'[a-zA-Z_][a-zA-Z0-9_]*'
    NUMBER = r'[0-9e.]+'
    ARGSEP = r','
    FUNCTIONSEP = r';'

    def error(self, t):
        print("Illegal character '%s'" % t.value[0])
        if t.value[0] == '{' and t.value.find('}'):
            self.index += 1 + t.value.find('}')
        else:
            self.index += 1


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
        return float(p.NUMBER)

    @_('IDENTIFIER')
    def arg(self, p):
        return p.IDENTIFIER


class TimedSequenceParser(Parser):
    tokens = TimedSequenceLexer.tokens

    @_('timedSequenceL', 'timedSequenceW')
    def timedSequence(self, p):
        return p[0]

    @_('loop')
    def timedSequenceL(self, p):
        return [p.loop]

    @_('loop timedSequenceW')
    def timedSequenceL(self, p):
        ret = [p.loop]
        ret.extend(p.timedSequenceW)
        return ret

    @_('timedWord')
    def timedSequenceW(self, p):
        return [p.timedWord]

    @_('timedWord timedSequenceL')
    def timedSequenceW(self, p):
        ret = [p.timedWord]
        ret.extend(p.timedSequenceL)
        return ret

    @_('timedSymbol FUNCTIONSEP timedWord')
    def timedWord(self, p):
        p.timedWord.word.insert(0, p.timedSymbol)
        return p.timedWord

    @_('timedSymbol FUNCTIONSEP')
    def timedWord(self, p):
        return TimedWord(word=[p.timedSymbol])

    @_('CYCLE LPAREN IDENTIFIER RPAREN LBRACE timedWord RBRACE')
    def loop(self, p):
        return Workload(p.IDENTIFIER, p.timedWord)

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
        return float(p.NUMBER)

    @_('IDENTIFIER')
    def arg(self, p):
        return p.IDENTIFIER

    def error(self, p):
        if p:
            print("Syntax error at token", p.type)
            # Just discard the token and tell the parser it's okay.
            self.errok()
        else:
            print("Syntax error at EOF")


class TimedWord:
    def __init__(self, word_string=None, word=list()):
        if word_string is not None:
            lexer = TimedWordLexer()
            parser = TimedWordParser()
            self.word = parser.parse(lexer.tokenize(word_string))
        else:
            self.word = word

    def __getitem__(self, item):
        return self.word[item]

    def __repr__(self):
        ret = list()
        for symbol in self.word:
            ret.append('{}({})'.format(symbol[0], ', '.join(map(str, symbol[1:]))))
        return 'TimedWord<"{}">'.format('; '.join(ret))


class Workload:
    def __init__(self, name, word):
        self.name = name
        self.word = word

    def __getitem__(self, item):
        return self.word[item]

    def __repr__(self):
        return 'Workload("{}", {})'.format(self.name, self.word)


class TimedSequence:
    def __init__(self, seq_string=None, seq=list()):
        if seq_string is not None:
            lexer = TimedSequenceLexer()
            parser = TimedSequenceParser()
            self.seq = parser.parse(lexer.tokenize(seq_string))
        else:
            self.seq = seq

    def __getitem__(self, item):
        return self.seq[item]

    def __repr__(self):
        ret = list()
        for symbol in self.seq:
            ret.append('{}'.format(symbol))
        return 'TimedSequence(seq=[{}])'.format(', '.join(ret))
