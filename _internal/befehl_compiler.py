#!/usr/bin/env python3
"""
Befehl Language Compiler v3
A complete compiler for the Befehl language that translates to Python code.

Language designed by Cwzx.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Optional, Union, List, Tuple
from enum import Enum, auto


# ============================================================================
# Token Types
# ============================================================================

import math


class TokenType(Enum):
    # Literals
    NUMBER = auto()
    STRING = auto()
    IDENTIFIER = auto()
    
    # Keywords
    BEFEHL = auto()
    LET = auto()
    FUNCTION = auto()
    CLASS = auto()
    WITH = auto()
    NAME = auto()
    INITIALIZE = auto()
    INHERIT = auto()
    EXCUTE = auto()
    IF = auto()
    ELIF = auto()
    ELSE = auto()
    WHILE = auto()
    FOREACH = auto()
    WHEREAS = auto()
    RETURN = auto()
    PRINT = auto()
    LAMBDA = auto()
    EXPORT = auto()
    FROM = auto()
    AS = auto()
    TYPE = auto()
    CAPABLE = auto()
    SUBCASE = auto()
    VALID = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    IN = auto()
    IS = auto()
    
    # Operators
    ARROW = auto()           # -> or |->
    ASSIGN = auto()          # :=
    LARROW = auto()          # <-
    PIPE = auto()            # |
    DOUBLE_PIPE = auto()     # ||
    COLON = auto()           # :
    SEMICOLON = auto()       # ;
    COMMA = auto()           # ,
    DOT = auto()             # .
    
    # Comparison operators
    EQ = auto()              # ==
    NE = auto()              # !=
    LT = auto()              # <
    GT = auto()              # >
    LE = auto()              # <=
    GE = auto()              # >=
    
    # Arithmetic operators
    PLUS = auto()            # +
    MINUS = auto()           # -
    STAR = auto()            # *
    SLASH = auto()           # /
    DSLASH = auto()          # //
    PERCENT = auto()         # %
    
    # Brackets
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    
    # Special
    FSTRING = auto()
    NEWLINE = auto()
    EOF = auto()
    PYTHON_BLOCK = auto()


@dataclass
class Token:
    type: TokenType
    value: Any
    line: int
    column: int
    
    def __repr__(self):
        return f"Token({self.type.name}, {self.value!r}, line={self.line})"


# ============================================================================
# Lexer
# ============================================================================

class Lexer:
    """Tokenizes Befehl source code."""
    
    KEYWORDS = {
        'Befehl': TokenType.BEFEHL,
        'let': TokenType.LET,
        'function': TokenType.FUNCTION,
        'class': TokenType.CLASS,
        'with': TokenType.WITH,
        'name': TokenType.NAME,
        'initialize': TokenType.INITIALIZE,
        'inherit': TokenType.INHERIT,
        'excute': TokenType.EXCUTE,
        'if': TokenType.IF,
        'elif': TokenType.ELIF,
        'else': TokenType.ELSE,
        'while': TokenType.WHILE,
        'foreach': TokenType.FOREACH,
        'whereas': TokenType.WHEREAS,
        'return': TokenType.RETURN,
        'print': TokenType.PRINT,
        'Lambda': TokenType.LAMBDA,
        'Export': TokenType.EXPORT,
        'from': TokenType.FROM,
        'as': TokenType.AS,
        'type': TokenType.TYPE,
        'capable': TokenType.CAPABLE,
        'subcase': TokenType.SUBCASE,
        'valid': TokenType.VALID,
        'and': TokenType.AND,
        'or': TokenType.OR,
        'not': TokenType.NOT,
        'in': TokenType.IN,
        'is': TokenType.IS,
    }
    
    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []
        
    def error(self, msg: str):
        raise SyntaxError(f"Lexer error at line {self.line}, column {self.column}: {msg}")
    
    def peek(self, offset: int = 0) -> str:
        pos = self.pos + offset
        if pos >= len(self.source):
            return '\0'
        return self.source[pos]
    
    def advance(self) -> str:
        char = self.peek()
        self.pos += 1
        if char == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return char
    
    def skip_whitespace(self):
        while self.peek() in ' \t\r':
            self.advance()
    
    def skip_comment(self) -> bool:
        if self.peek() == '/' and self.peek(1) == '/':
            while self.peek() != '\n' and self.peek() != '\0':
                self.advance()
            return True
        elif self.peek() == '/' and self.peek(1) == '*':
            self.advance()
            self.advance()
            while not (self.peek() == '*' and self.peek(1) == '/'):
                if self.peek() == '\0':
                    self.error("Unterminated multi-line comment")
                self.advance()
            self.advance()
            self.advance()
            return True
        return False
    
    def read_string(self, quote: str) -> str:
        """Read a string with the given quote character."""
        result = []
        self.advance()  # Skip opening quote
        
        while self.peek() != quote:
            if self.peek() == '\0':
                self.error("Unterminated string")
            if self.peek() == '\\':
                self.advance()
                escape_char = self.advance()
                escape_map = {'n': '\n', 't': '\t', 'r': '\r', '\\': '\\', "'": "'", '"': '"', '`': '`'}
                result.append(escape_map.get(escape_char, escape_char))
            else:
                result.append(self.advance())
        
        self.advance()  # Skip closing quote
        return ''.join(result)
    
    def read_fstring(self) -> str:
        """Read an f-string like f`...' or f\"...\"."""
        self.advance()  # Skip 'f'
        quote = self.advance()  # Get the quote character
        result = []
        
        while self.peek() != quote:
            if self.peek() == '\0':
                self.error("Unterminated f-string")
            if self.peek() == '\\':
                self.advance()
                escape_char = self.advance()
                escape_map = {'n': '\n', 't': '\t', 'r': '\r', '\\': '\\', "'": "'", '"': '"', '`': '`'}
                result.append(escape_map.get(escape_char, escape_char))
            else:
                result.append(self.advance())
        
        self.advance()  # Skip closing quote
        return ''.join(result)
    
    def read_number(self) -> Union[int, float]:
        """Read a number literal."""
        result = []
        has_dot = False
        
        while self.peek().isdigit() or (self.peek() == '.' and not has_dot):
            if self.peek() == '.':
                if self.peek(1) == '.':
                    break  # .. operator
                has_dot = True
            result.append(self.advance())
        
        num_str = ''.join(result)
        return float(num_str) if has_dot else int(num_str)
    
    def read_identifier(self) -> str:
        """Read an identifier or keyword."""
        result = []
        while self.peek().isalnum() or self.peek() == '_':
            result.append(self.advance())
        return ''.join(result)
    
    def read_python_block(self) -> str:
        """Read a Python code block between ``` or '''."""
        start = self.source[self.pos:self.pos+3]
        self.advance()
        self.advance()
        self.advance()
        
        result = []
        while True:
            if self.peek() == '\0':
                self.error("Unterminated Python block")
            curr = self.source[self.pos:self.pos+3]
            if curr == start:
                self.advance()
                self.advance()
                self.advance()
                break
            result.append(self.advance())
        
        return ''.join(result).strip()
    
    def tokenize(self) -> List[Token]:
        """Tokenize the source code."""
        while self.pos < len(self.source):
            self.skip_whitespace()
            
            if self.peek() == '\0':
                break
                
            line, column = self.line, self.column
            
            # Comments
            if self.skip_comment():
                continue
            
            # Newline
            if self.peek() == '\n':
                self.advance()
                self.tokens.append(Token(TokenType.NEWLINE, '\n', line, column))
                continue
            
            # Python block
            if (self.peek() == '`' and self.peek(1) == '`' and self.peek(2) == '`') or \
               (self.peek() == "'" and self.peek(1) == "'" and self.peek(2) == "'"):
                block = self.read_python_block()
                self.tokens.append(Token(TokenType.PYTHON_BLOCK, block, line, column))
                continue
            
            # F-string: f`...' or f"..." or f'...'
            if self.peek() == 'f' and self.peek(1) in '`"\'':
                fstring = self.read_fstring()
                self.tokens.append(Token(TokenType.FSTRING, fstring, line, column))
                continue
            
            # Strings (backtick, double quote, single quote)
            if self.peek() == '`':
                string = self.read_string('`')
                self.tokens.append(Token(TokenType.STRING, string, line, column))
                continue
            
            if self.peek() in '"\'':
                string = self.read_string(self.peek())
                self.tokens.append(Token(TokenType.STRING, string, line, column))
                continue
            
            # Numbers
            if self.peek().isdigit():
                num = self.read_number()
                self.tokens.append(Token(TokenType.NUMBER, num, line, column))
                continue
            
            # Multi-character operators
            if self.peek() == '-' and self.peek(1) == '>':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.ARROW, '->', line, column))
                continue
            
            if self.peek() == ':' and self.peek(1) == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.ASSIGN, ':=', line, column))
                continue
            
            if self.peek() == '<' and self.peek(1) == '-':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.LARROW, '<-', line, column))
                continue
            
            if self.peek() == '|' and self.peek(1) == '|':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.DOUBLE_PIPE, '||', line, column))
                continue
            
            if self.peek() == '|' and self.peek(1) == '-' and self.peek(2) == '>':
                self.advance()
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.ARROW, '|->', line, column))
                continue
            
            if self.peek() == '=' and self.peek(1) == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.EQ, '==', line, column))
                continue
            
            if self.peek() == '!' and self.peek(1) == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.NE, '!=', line, column))
                continue
            
            if self.peek() == '<' and self.peek(1) == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.LE, '<=', line, column))
                continue
            
            if self.peek() == '>' and self.peek(1) == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.GE, '>=', line, column))
                continue
            
            if self.peek() == '/' and self.peek(1) == '/':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.DSLASH, '//', line, column))
                continue
            
            # Single character tokens
            single_chars = {
                ':': TokenType.COLON,
                ';': TokenType.SEMICOLON,
                ',': TokenType.COMMA,
                '.': TokenType.DOT,
                '|': TokenType.PIPE,
                '(': TokenType.LPAREN,
                ')': TokenType.RPAREN,
                '{': TokenType.LBRACE,
                '}': TokenType.RBRACE,
                '[': TokenType.LBRACKET,
                ']': TokenType.RBRACKET,
                '+': TokenType.PLUS,
                '-': TokenType.MINUS,
                '*': TokenType.STAR,
                '/': TokenType.SLASH,
                '%': TokenType.PERCENT,
                '<': TokenType.LT,
                '>': TokenType.GT,
                '=': TokenType.ASSIGN,
            }
            
            if self.peek() in single_chars:
                char = self.advance()
                self.tokens.append(Token(single_chars[char], char, line, column))
                continue
            
            # Identifiers and keywords
            if self.peek().isalpha() or self.peek() == '_':
                ident = self.read_identifier()
                
                # Check for Befehl: keyword
                if ident == 'Befehl' and self.peek() == ':':
                    self.advance()
                    self.tokens.append(Token(TokenType.BEFEHL, 'Befehl:', line, column))
                else:
                    token_type = self.KEYWORDS.get(ident, TokenType.IDENTIFIER)
                    self.tokens.append(Token(token_type, ident, line, column))
                continue
            
            # Unknown character - skip it
            char = self.advance()
        
        self.tokens.append(Token(TokenType.EOF, None, self.line, self.column))
        return self.tokens


# ============================================================================
# AST Nodes
# ============================================================================

@dataclass
class ASTNode:
    """Base class for AST nodes."""
    pass


@dataclass
class Program(ASTNode):
    statements: List[ASTNode] = field(default_factory=list)


@dataclass
class BefehlStatement(ASTNode):
    statement: ASTNode


@dataclass
class VariableDeclaration(ASTNode):
    name: str
    value: ASTNode
    type_annotation: Optional[str] = None


@dataclass 
class Assignment(ASTNode):
    targets: List[ASTNode]
    value: ASTNode


@dataclass
class NumberLiteral(ASTNode):
    value: Union[int, float]


@dataclass
class StringLiteral(ASTNode):
    value: str


@dataclass
class FString(ASTNode):
    template: str


@dataclass
class Identifier(ASTNode):
    name: str


@dataclass
class BinaryOp(ASTNode):
    left: ASTNode
    op: str
    right: ASTNode


@dataclass
class UnaryOp(ASTNode):
    op: str
    operand: ASTNode


@dataclass
class FunctionCall(ASTNode):
    function: ASTNode
    arguments: List[ASTNode] = field(default_factory=list)


@dataclass
class MemberAccess(ASTNode):
    object: ASTNode
    member: str


@dataclass
class FunctionDefinition(ASTNode):
    name: str
    parameters: List[Tuple[str, Optional[str]]]
    body: List[ASTNode]
    return_type: Optional[str] = None


@dataclass
class LambdaExpression(ASTNode):
    parameters: List[str]
    body: ASTNode
    type_annotation: Optional[str] = None


@dataclass
class ClassDefinition(ASTNode):
    name: str
    base_class: Optional[str] = None
    methods: List[ASTNode] = field(default_factory=list)


@dataclass
class MethodDefinition(ASTNode):
    name: str
    parameters: List[Tuple[str, Optional[str]]]
    body: List[ASTNode]
    is_initialize: bool = False
    is_inherit: bool = False


@dataclass
class IfStatement(ASTNode):
    condition: ASTNode
    then_body: List[ASTNode]
    elif_clauses: List[Tuple[ASTNode, List[ASTNode]]] = field(default_factory=list)
    else_body: Optional[List[ASTNode]] = None


@dataclass
class WhileStatement(ASTNode):
    condition: ASTNode
    body: List[ASTNode]


@dataclass
class ForEachStatement(ASTNode):
    variable: Optional[str]
    iterable: Optional[ASTNode]
    whereas_clause: Optional[ASTNode]
    body: List[ASTNode]


@dataclass
class ReturnStatement(ASTNode):
    value: Optional[ASTNode] = None


@dataclass
class PrintStatement(ASTNode):
    value: ASTNode


@dataclass
class ImportStatement(ASTNode):
    module: str
    names: Optional[List[str]] = None
    as_name: Optional[str] = None
    use_import_func: bool = False


@dataclass
class PythonBlock(ASTNode):
    code: str


@dataclass
class ListLiteral(ASTNode):
    elements: List[ASTNode] = field(default_factory=list)


@dataclass
class DictLiteral(ASTNode):
    pairs: List[Tuple[ASTNode, ASTNode]] = field(default_factory=list)


@dataclass
class SetLiteral(ASTNode):
    elements: List[ASTNode] = field(default_factory=list)


@dataclass
class IndexAccess(ASTNode):
    object: ASTNode
    index: ASTNode


@dataclass
class TupleLiteral(ASTNode):
    elements: List[ASTNode] = field(default_factory=list)


@dataclass
class CapableCheck(ASTNode):
    statement: ASTNode


@dataclass
class ExpressionStatement(ASTNode):
    expression: ASTNode


# ============================================================================
# Parser
# ============================================================================

class Parser:
    """Parses Befehl tokens into an AST."""
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        
    def error(self, msg: str):
        token = self.current()
        raise SyntaxError(f"Parser error at line {token.line}: {msg}")
    
    def current(self) -> Token:
        if self.pos >= len(self.tokens):
            return self.tokens[-1]
        return self.tokens[self.pos]
    
    def peek(self, offset: int = 0) -> Token:
        pos = self.pos + offset
        if pos >= len(self.tokens):
            return self.tokens[-1]
        return self.tokens[pos]
    
    def advance(self) -> Token:
        token = self.current()
        self.pos += 1
        return token
    
    def match(self, *types: TokenType) -> bool:
        return self.current().type in types
    
    def match_identifier(self) -> bool:
        """Check if current token is an identifier or a keyword that can be used as identifier."""
        if self.match(TokenType.IDENTIFIER):
            return True
        # These keywords can also be used as identifiers in expressions
        if self.match(TokenType.NAME, TokenType.TYPE, TokenType.AS, TokenType.FROM,
                      TokenType.INITIALIZE, TokenType.INHERIT, TokenType.CAPABLE,
                      TokenType.VALID, TokenType.SUBCASE, TokenType.CLASS,
                      TokenType.FUNCTION, TokenType.WHILE, TokenType.IF,
                      TokenType.ELIF, TokenType.ELSE, TokenType.FOREACH,
                      TokenType.WHEREAS, TokenType.RETURN, TokenType.LAMBDA,
                      TokenType.EXPORT, TokenType.AND, TokenType.OR, TokenType.NOT,
                      TokenType.IN, TokenType.IS):
            return True
        return False
    
    def expect(self, token_type: TokenType, msg: str = None) -> Token:
        if not self.match(token_type):
            self.error(msg or f"Expected {token_type.name}, got {self.current().type.name}")
        return self.advance()
    
    def skip_newlines(self):
        while self.match(TokenType.NEWLINE):
            self.advance()
    
    def parse(self) -> Program:
        """Parse the entire program."""
        program = Program()
        self.skip_newlines()
        
        while not self.match(TokenType.EOF):
            stmt = self.parse_statement()
            if stmt:
                program.statements.append(stmt)
            self.skip_newlines()
        
        return program
    
    def parse_statement(self) -> Optional[ASTNode]:
        """Parse a single statement."""
        self.skip_newlines()
        
        # Befehl: marked statement
        if self.match(TokenType.BEFEHL):
            return self.parse_befehl_statement()
        
        # Non-Befehl statements
        if self.match(TokenType.EXCUTE):
            return self.parse_excute_statement()
        
        if self.match(TokenType.PRINT):
            return self.parse_print_statement()
        
        if self.match(TokenType.RETURN):
            return self.parse_return_statement()
        
        # Expression or assignment
        return self.parse_expression_statement()
    
    def parse_befehl_statement(self) -> BefehlStatement:
        """Parse a Befehl: marked statement."""
        self.expect(TokenType.BEFEHL)
        self.skip_newlines()
        
        stmt = self.parse_inner_statement()
        return BefehlStatement(stmt)
    
    def parse_inner_statement(self) -> ASTNode:
        """Parse a statement inside Befehl: block."""
        self.skip_newlines()
        
        # let statement
        if self.match(TokenType.LET):
            return self.parse_let_statement()
        
        # excute statement (for if/while/foreach/import inside Befehl:)
        if self.match(TokenType.EXCUTE):
            return self.parse_excute_statement()
        
        # Assignment or expression starting with identifier or ( for special names
        if self.match_identifier() or self.match(TokenType.LPAREN):
            return self.parse_assignment_or_expression()
        
        # Expression starting with other tokens
        return self.parse_expression()
    
    def parse_let_statement(self) -> ASTNode:
        """Parse let statement for variables, functions, or classes."""
        self.expect(TokenType.LET)
        self.skip_newlines()
        
        # let class with name `Person' := { ... }
        if self.match(TokenType.CLASS):
            return self.parse_class_definition()
        
        # let function with name `f' <- params { ... }
        if self.match(TokenType.FUNCTION):
            return self.parse_function_definition()
        
        # let value -> name (variable declaration)
        return self.parse_variable_declaration()
    
    def parse_variable_declaration(self) -> VariableDeclaration:
        """Parse variable declaration: let value -> name."""
        value = self.parse_expression()
        
        self.skip_newlines()
        self.expect(TokenType.ARROW, "Expected '->' in variable declaration")
        self.skip_newlines()
        
        name = self.parse_identifier_name()
        
        return VariableDeclaration(name=name, value=value)
    
    def parse_identifier_name(self) -> str:
        """Parse an identifier name (can be in backticks or normal)."""
        if self.match(TokenType.STRING):
            return self.advance().value
        if self.match_identifier():
            name = self.advance().value
            # Check for member access like self.name
            if self.match(TokenType.DOT):
                self.advance()
                member = self.parse_identifier_name()
                return name + '.' + member
            return name
        if self.match(TokenType.LPAREN):
            # Special case: (int) as variable name
            self.advance()
            parts = []
            while not self.match(TokenType.RPAREN):
                token = self.advance()
                parts.append(str(token.value))
            self.expect(TokenType.RPAREN)
            return '(' + ''.join(parts) + ')'
        self.error("Expected identifier name")
    
    def parse_assignment_or_expression(self) -> ASTNode:
        """Parse assignment or expression."""
        # Parse the left side (could be multiple targets separated by commas)
        targets = []
        
        while True:
            expr = self.parse_postfix_expression()
            targets.append(expr)
            
            if self.match(TokenType.COMMA):
                self.advance()
                self.skip_newlines()
            else:
                break
        
        # Check if it's an assignment
        if self.match(TokenType.ASSIGN):
            self.advance()
            self.skip_newlines()
            # Parse the value - could be a tuple if there are multiple targets
            if len(targets) > 1:
                # Parse as tuple
                values = [self.parse_expression()]
                while self.match(TokenType.COMMA):
                    self.advance()
                    self.skip_newlines()
                    values.append(self.parse_expression())
                if len(values) == 1:
                    value = values[0]
                else:
                    value = TupleLiteral(elements=values)
            else:
                value = self.parse_expression()
            return Assignment(targets=targets, value=value)
        
        # Not an assignment, return as expression statement
        if len(targets) == 1:
            return ExpressionStatement(targets[0])
        return ExpressionStatement(TupleLiteral(elements=targets))
    
    def parse_class_definition(self) -> ClassDefinition:
        """Parse class definition."""
        self.expect(TokenType.CLASS)
        self.skip_newlines()
        self.expect(TokenType.WITH, "Expected 'with' after 'class'")
        self.skip_newlines()
        self.expect(TokenType.NAME, "Expected 'name' after 'with'")
        self.skip_newlines()
        
        name = self.parse_identifier_name()
        self.skip_newlines()
        
        # Check for inheritance
        base_class = None
        if self.match(TokenType.LPAREN):
            self.advance()
            base_class = self.parse_identifier_name()
            self.expect(TokenType.RPAREN)
            self.skip_newlines()
        
        self.expect(TokenType.ASSIGN, "Expected ':=' after class name")
        self.skip_newlines()
        self.expect(TokenType.LBRACE, "Expected '{' to start class body")
        self.skip_newlines()
        
        methods = []
        while not self.match(TokenType.RBRACE):
            if self.match(TokenType.LET):
                method = self.parse_method_definition()
                methods.append(method)
            elif self.match(TokenType.NEWLINE):
                self.skip_newlines()
            else:
                self.error(f"Unexpected token in class body: {self.current().type.name}")
        
        self.expect(TokenType.RBRACE)
        return ClassDefinition(name=name, base_class=base_class, methods=methods)
    
    def parse_method_definition(self) -> MethodDefinition:
        """Parse method definition within a class."""
        self.expect(TokenType.LET)
        self.skip_newlines()
        self.expect(TokenType.CLASS, "Expected 'class' in method definition")
        self.skip_newlines()
        
        is_initialize = False
        is_inherit = False
        
        if self.match(TokenType.INITIALIZE):
            self.advance()
            self.skip_newlines()
            is_initialize = True
        elif self.match(TokenType.INHERIT):
            self.advance()
            self.skip_newlines()
            is_inherit = True
        
        self.expect(TokenType.FUNCTION, "Expected 'function' in method definition")
        self.skip_newlines()
        self.expect(TokenType.WITH, "Expected 'with' in method definition")
        self.skip_newlines()
        self.expect(TokenType.NAME, "Expected 'name' in method definition")
        self.skip_newlines()
        
        name = self.parse_identifier_name()
        self.skip_newlines()
        
        parameters = []
        if self.match(TokenType.LARROW):
            self.advance()
            self.skip_newlines()
            parameters = self.parse_parameter_list()
            self.skip_newlines()
        
        self.expect(TokenType.LBRACE, "Expected '{' to start method body")
        self.skip_newlines()
        
        body = []
        while not self.match(TokenType.RBRACE):
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
            self.skip_newlines()
        
        self.expect(TokenType.RBRACE)
        
        return MethodDefinition(
            name=name,
            parameters=parameters,
            body=body,
            is_initialize=is_initialize,
            is_inherit=is_inherit
        )
    
    def parse_function_definition(self) -> FunctionDefinition:
        """Parse function definition."""
        self.expect(TokenType.FUNCTION)
        self.skip_newlines()
        self.expect(TokenType.WITH, "Expected 'with' after 'function'")
        self.skip_newlines()
        self.expect(TokenType.NAME, "Expected 'name' after 'with'")
        self.skip_newlines()
        
        name = self.parse_identifier_name()
        self.skip_newlines()
        
        parameters = []
        if self.match(TokenType.LARROW):
            self.advance()
            self.skip_newlines()
            parameters = self.parse_parameter_list()
            self.skip_newlines()
        
        self.expect(TokenType.LBRACE, "Expected '{' to start function body")
        self.skip_newlines()
        
        body = []
        while not self.match(TokenType.RBRACE):
            if self.match(TokenType.PYTHON_BLOCK):
                block = self.advance().value
                body.append(PythonBlock(code=block))
            else:
                stmt = self.parse_statement()
                if stmt:
                    body.append(stmt)
            self.skip_newlines()
        
        self.expect(TokenType.RBRACE)
        
        return FunctionDefinition(name=name, parameters=parameters, body=body)
    
    def parse_parameter_list(self) -> List[Tuple[str, Optional[str]]]:
        """Parse parameter list."""
        parameters = []
        
        while not self.match(TokenType.LBRACE) and not self.match(TokenType.NEWLINE):
            param_name = self.parse_identifier_name()
            param_type = None
            
            if self.match(TokenType.LPAREN):
                self.advance()
                type_parts = []
                while not self.match(TokenType.RPAREN):
                    type_parts.append(str(self.advance().value))
                self.expect(TokenType.RPAREN)
                param_type = ''.join(type_parts)
            
            parameters.append((param_name, param_type))
            
            if self.match(TokenType.COMMA):
                self.advance()
                self.skip_newlines()
            else:
                break
        
        return parameters
    
    def parse_excute_statement(self) -> ASTNode:
        """Parse excute statement (if/while/foreach/import)."""
        self.expect(TokenType.EXCUTE)
        self.skip_newlines()
        
        # excute if capable: (Export ...)
        if self.match(TokenType.IF):
            self.advance()
            self.skip_newlines()
            
            if self.match(TokenType.CAPABLE):
                self.advance()
                self.skip_newlines()
                self.expect(TokenType.COLON)
                self.skip_newlines()
                
                self.expect(TokenType.LPAREN)
                self.skip_newlines()
                
                import_stmt = self.parse_export_statement()
                
                self.skip_newlines()
                self.expect(TokenType.RPAREN)
                
                return CapableCheck(statement=import_stmt)
            
            # Regular if statement
            condition = self.parse_expression()
            self.skip_newlines()
            self.expect(TokenType.LBRACE)
            self.skip_newlines()
            
            then_body = []
            while not self.match(TokenType.RBRACE):
                stmt = self.parse_statement()
                if stmt:
                    then_body.append(stmt)
                self.skip_newlines()
            
            self.expect(TokenType.RBRACE)
            
            # Parse elif clauses
            elif_clauses = []
            while self.match(TokenType.ELIF):
                self.advance()
                self.skip_newlines()
                elif_cond = self.parse_expression()
                self.skip_newlines()
                self.expect(TokenType.LBRACE)
                self.skip_newlines()
                
                elif_body = []
                while not self.match(TokenType.RBRACE):
                    stmt = self.parse_statement()
                    if stmt:
                        elif_body.append(stmt)
                    self.skip_newlines()
                
                self.expect(TokenType.RBRACE)
                elif_clauses.append((elif_cond, elif_body))
            
            # Parse else clause
            else_body = None
            if self.match(TokenType.ELSE):
                self.advance()
                self.skip_newlines()
                self.expect(TokenType.LBRACE)
                self.skip_newlines()
                
                else_body = []
                while not self.match(TokenType.RBRACE):
                    stmt = self.parse_statement()
                    if stmt:
                        else_body.append(stmt)
                    self.skip_newlines()
                
                self.expect(TokenType.RBRACE)
            
            return IfStatement(
                condition=condition,
                then_body=then_body,
                elif_clauses=elif_clauses,
                else_body=else_body
            )
        
        # excute while condition { }
        if self.match(TokenType.WHILE):
            self.advance()
            self.skip_newlines()
            condition = self.parse_expression()
            self.skip_newlines()
            self.expect(TokenType.LBRACE)
            self.skip_newlines()
            
            body = []
            while not self.match(TokenType.RBRACE):
                stmt = self.parse_statement()
                if stmt:
                    body.append(stmt)
                self.skip_newlines()
            
            self.expect(TokenType.RBRACE)
            return WhileStatement(condition=condition, body=body)
        
        # excute foreach ... { }
        if self.match(TokenType.FOREACH):
            return self.parse_foreach_statement()
        
        # excute with ... foreach ... { }
        if self.match(TokenType.WITH):
            self.advance()
            self.skip_newlines()
            lambda_expr = self.parse_expression()
            self.skip_newlines()
            return self.parse_foreach_statement(lambda_expr)
        
        self.error(f"Unexpected token after 'excute': {self.current().type.name}")
    
    def parse_foreach_statement(self, lambda_expr: ASTNode = None) -> ForEachStatement:
        """Parse foreach statement."""
        self.expect(TokenType.FOREACH)
        self.skip_newlines()
        
        variable = None
        iterable = lambda_expr
        
        if self.match(TokenType.LPAREN):
            self.advance()
            self.skip_newlines()
            
            if self.match(TokenType.IDENTIFIER):
                var_name = self.advance().value
                if self.match(TokenType.ASSIGN):
                    self.advance()
                    self.skip_newlines()
                    variable = var_name
                    iterable = self.parse_expression()
                else:
                    self.pos -= 1
                    iterable = self.parse_expression()
            
            self.skip_newlines()
            self.expect(TokenType.RPAREN)
            self.skip_newlines()
        
        whereas_clause = None
        if self.match(TokenType.WHEREAS):
            self.advance()
            self.skip_newlines()
            self.expect(TokenType.LBRACKET)
            self.skip_newlines()
            whereas_clause = self.parse_expression()
            self.skip_newlines()
            self.expect(TokenType.RBRACKET)
            self.skip_newlines()
        
        self.expect(TokenType.LBRACE)
        self.skip_newlines()
        
        body = []
        while not self.match(TokenType.RBRACE):
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
            self.skip_newlines()
        
        self.expect(TokenType.RBRACE)
        
        return ForEachStatement(
            variable=variable,
            iterable=iterable,
            whereas_clause=whereas_clause,
            body=body
        )
    
    def parse_export_statement(self) -> ImportStatement:
        """Parse Export statement (import)."""
        self.expect(TokenType.EXPORT)
        self.skip_newlines()
        
        module = self.parse_identifier_name()
        self.skip_newlines()
        
        names = None
        as_name = None
        use_import_func = False
        
        if self.match(TokenType.FROM):
            self.advance()
            self.skip_newlines()
            actual_module = self.parse_identifier_name()
            self.skip_newlines()
            names = [module]
            module = actual_module
        
        if self.match(TokenType.AS):
            self.advance()
            self.skip_newlines()
            as_name = self.parse_identifier_name()
            self.skip_newlines()
        
        if self.match(TokenType.TYPE):
            self.advance()
            self.skip_newlines()
            if self.match(TokenType.IDENTIFIER) and self.current().value == '__import__':
                self.advance()
                use_import_func = True
        
        return ImportStatement(
            module=module,
            names=names,
            as_name=as_name,
            use_import_func=use_import_func
        )
    
    def parse_print_statement(self) -> PrintStatement:
        """Parse print statement."""
        self.expect(TokenType.PRINT)
        self.skip_newlines()
        self.expect(TokenType.LPAREN)
        self.skip_newlines()
        
        value = self.parse_expression()
        
        self.skip_newlines()
        self.expect(TokenType.RPAREN)
        
        return PrintStatement(value=value)
    
    def parse_return_statement(self) -> ReturnStatement:
        """Parse return statement."""
        self.expect(TokenType.RETURN)
        self.skip_newlines()
        
        value = None
        if not self.match(TokenType.SEMICOLON) and not self.match(TokenType.RBRACE) and not self.match(TokenType.NEWLINE):
            value = self.parse_expression()
        
        return ReturnStatement(value=value)
    
    def parse_expression_statement(self) -> ASTNode:
        """Parse an expression as a statement."""
        expr = self.parse_expression()
        return ExpressionStatement(expr)
    
    def parse_expression(self) -> ASTNode:
        """Parse an expression."""
        return self.parse_or_expression()
    
    def parse_or_expression(self) -> ASTNode:
        """Parse or expression."""
        left = self.parse_and_expression()
        
        while self.match(TokenType.OR) or self.match(TokenType.DOUBLE_PIPE):
            self.advance()
            self.skip_newlines()
            right = self.parse_and_expression()
            left = BinaryOp(left=left, op='or', right=right)
        
        return left
    
    def parse_and_expression(self) -> ASTNode:
        """Parse and expression."""
        left = self.parse_not_expression()
        
        while self.match(TokenType.AND):
            self.advance()
            self.skip_newlines()
            right = self.parse_not_expression()
            left = BinaryOp(left=left, op='and', right=right)
        
        return left
    
    def parse_not_expression(self) -> ASTNode:
        """Parse not expression."""
        if self.match(TokenType.NOT):
            self.advance()
            self.skip_newlines()
            operand = self.parse_not_expression()
            return UnaryOp(op='not', operand=operand)
        
        return self.parse_comparison_expression()
    
    def parse_comparison_expression(self) -> ASTNode:
        """Parse comparison expression."""
        left = self.parse_additive_expression()
        
        while True:
            op = None
            
            if self.match(TokenType.EQ):
                op = '=='
                self.advance()
            elif self.match(TokenType.NE):
                op = '!='
                self.advance()
            elif self.match(TokenType.LT):
                op = '<'
                self.advance()
            elif self.match(TokenType.GT):
                op = '>'
                self.advance()
            elif self.match(TokenType.LE):
                op = '<='
                self.advance()
            elif self.match(TokenType.GE):
                op = '>='
                self.advance()
            elif self.match(TokenType.IS):
                op = 'is'
                self.advance()
            elif self.match(TokenType.IN):
                op = 'in'
                self.advance()
            elif self.match(TokenType.NOT):
                self.advance()
                if self.match(TokenType.IN):
                    op = 'not in'
                    self.advance()
            
            if op:
                self.skip_newlines()
                right = self.parse_additive_expression()
                left = BinaryOp(left=left, op=op, right=right)
            else:
                break
        
        return left
    
    def parse_additive_expression(self) -> ASTNode:
        """Parse additive expression (+, -)."""
        left = self.parse_multiplicative_expression()
        
        while self.match(TokenType.PLUS) or self.match(TokenType.MINUS):
            op = '+' if self.match(TokenType.PLUS) else '-'
            self.advance()
            self.skip_newlines()
            right = self.parse_multiplicative_expression()
            left = BinaryOp(left=left, op=op, right=right)
        
        return left
    
    def parse_multiplicative_expression(self) -> ASTNode:
        """Parse multiplicative expression (*, /, //, %)."""
        left = self.parse_unary_expression()
        
        while self.match(TokenType.STAR) or self.match(TokenType.SLASH) or \
              self.match(TokenType.DSLASH) or self.match(TokenType.PERCENT):
            if self.match(TokenType.STAR):
                op = '*'
            elif self.match(TokenType.SLASH):
                op = '/'
            elif self.match(TokenType.DSLASH):
                op = '//'
            else:
                op = '%'
            self.advance()
            self.skip_newlines()
            right = self.parse_unary_expression()
            left = BinaryOp(left=left, op=op, right=right)
        
        return left
    
    def parse_unary_expression(self) -> ASTNode:
        """Parse unary expression."""
        if self.match(TokenType.MINUS):
            self.advance()
            self.skip_newlines()
            operand = self.parse_unary_expression()
            return UnaryOp(op='-', operand=operand)
        
        if self.match(TokenType.PLUS):
            self.advance()
            self.skip_newlines()
            return self.parse_unary_expression()
        
        return self.parse_postfix_expression()
    
    def parse_postfix_expression(self) -> ASTNode:
        """Parse postfix expression (function call, member access, index)."""
        expr = self.parse_primary_expression()
        
        while True:
            if self.match(TokenType.LPAREN):
                self.advance()
                self.skip_newlines()
                args = []
                while not self.match(TokenType.RPAREN):
                    args.append(self.parse_expression())
                    if self.match(TokenType.COMMA):
                        self.advance()
                        self.skip_newlines()
                self.expect(TokenType.RPAREN)
                expr = FunctionCall(function=expr, arguments=args)
            elif self.match(TokenType.DOT):
                self.advance()
                member = self.parse_identifier_name()
                expr = MemberAccess(object=expr, member=member)
            elif self.match(TokenType.LBRACKET):
                self.advance()
                self.skip_newlines()
                index = self.parse_expression()
                self.skip_newlines()
                self.expect(TokenType.RBRACKET)
                expr = IndexAccess(object=expr, index=index)
            elif self.match(TokenType.DOUBLE_PIPE):
                # args || func means func(args)
                self.advance()
                self.skip_newlines()
                func = self.parse_primary_expression()
                if isinstance(expr, TupleLiteral):
                    args = expr.elements
                else:
                    args = [expr]
                expr = FunctionCall(function=func, arguments=args)
            else:
                break
        
        return expr
    
    def parse_primary_expression(self) -> ASTNode:
        """Parse primary expression."""
        # Number
        if self.match(TokenType.NUMBER):
            return NumberLiteral(value=self.advance().value)
        
        # String
        if self.match(TokenType.STRING):
            return StringLiteral(value=self.advance().value)
        
        # F-string
        if self.match(TokenType.FSTRING):
            return FString(template=self.advance().value)
        
        # Lambda keyword
        if self.match(TokenType.LAMBDA):
            return self.parse_lambda_expression()
        
        # Python block
        if self.match(TokenType.PYTHON_BLOCK):
            return PythonBlock(code=self.advance().value)
        
        # Befehl: expression (inline)
        if self.match(TokenType.BEFEHL):
            self.advance()
            self.skip_newlines()
            expr = self.parse_expression()
            return expr
        
        # Parenthesized expression, tuple, or special identifier
        if self.match(TokenType.LPAREN):
            self.advance()
            self.skip_newlines()
            
            # Check for special identifier like (Lambda) or (int)
            if self.match(TokenType.LAMBDA):
                # (Lambda) as type annotation
                self.advance()
                self.skip_newlines()
                self.expect(TokenType.RPAREN)
                # Return a marker for Lambda type
                return Identifier(name='(Lambda)')
            
            if self.match(TokenType.IDENTIFIER):
                # Could be (int) or (type) as special identifier
                next_tok = self.peek(1)
                if next_tok.type == TokenType.RPAREN:
                    # It's a special identifier like (int)
                    name = '(' + self.advance().value + ')'
                    self.expect(TokenType.RPAREN)
                    return Identifier(name=name)
            
            if self.match(TokenType.RPAREN):
                self.advance()
                return TupleLiteral(elements=[])
            
            expr = self.parse_expression()
            
            if self.match(TokenType.COMMA):
                elements = [expr]
                while self.match(TokenType.COMMA):
                    self.advance()
                    self.skip_newlines()
                    if self.match(TokenType.RPAREN):
                        break
                    elements.append(self.parse_expression())
                self.expect(TokenType.RPAREN)
                return TupleLiteral(elements=elements)
            
            self.expect(TokenType.RPAREN)
            return expr
        
        # List
        if self.match(TokenType.LBRACKET):
            self.advance()
            self.skip_newlines()
            
            elements = []
            while not self.match(TokenType.RBRACKET):
                elements.append(self.parse_expression())
                if self.match(TokenType.COMMA):
                    self.advance()
                    self.skip_newlines()
            
            self.expect(TokenType.RBRACKET)
            return ListLiteral(elements=elements)
        
        # Set/Dict
        if self.match(TokenType.LBRACE):
            self.advance()
            self.skip_newlines()
            
            if self.match(TokenType.RBRACE):
                self.advance()
                return SetLiteral(elements=[])
            
            first = self.parse_expression()
            
            if self.match(TokenType.COLON):
                self.advance()
                self.skip_newlines()
                value = self.parse_expression()
                pairs = [(first, value)]
                
                while self.match(TokenType.COMMA):
                    self.advance()
                    self.skip_newlines()
                    if self.match(TokenType.RBRACE):
                        break
                    key = self.parse_expression()
                    self.expect(TokenType.COLON)
                    self.skip_newlines()
                    val = self.parse_expression()
                    pairs.append((key, val))
                
                self.expect(TokenType.RBRACE)
                return DictLiteral(pairs=pairs)
            else:
                elements = [first]
                while self.match(TokenType.COMMA):
                    self.advance()
                    self.skip_newlines()
                    if self.match(TokenType.RBRACE):
                        break
                    elements.append(self.parse_expression())
                
                self.expect(TokenType.RBRACE)
                return SetLiteral(elements=elements)
        
        # Identifier
        if self.match_identifier():
            name = self.advance().value
            
            # Check for lambda: x |-> expr
            if self.match(TokenType.ARROW) and self.current().value == '|->':
                self.advance()
                self.skip_newlines()
                body = self.parse_expression()
                return LambdaExpression(parameters=[name], body=body)
            
            return Identifier(name=name)
        
        self.error(f"Unexpected token in expression: {self.current().type.name} ({self.current().value})")
    
    def parse_lambda_expression(self) -> LambdaExpression:
        """Parse lambda expression: Lambda: type, params |-> body."""
        self.expect(TokenType.LAMBDA)
        self.skip_newlines()
        
        type_annotation = None
        
        # Check for colon (type annotation)
        if self.match(TokenType.COLON):
            self.advance()
            self.skip_newlines()
            # Parse type annotation (e.g., int -> int)
            # We need to be careful not to consume the |-> arrow
            type_parts = []
            while not self.match(TokenType.COMMA) and not self.match(TokenType.EOF):
                # Check if this is the |-> arrow (lambda body arrow)
                if self.match(TokenType.ARROW) and self.current().value == '|->':
                    break
                # Check if next token after -> is a parameter name followed by |->
                if self.match(TokenType.ARROW) and self.peek(1).type in [TokenType.IDENTIFIER, TokenType.COMMA]:
                    # This might be the end of type annotation
                    if self.peek(1).type == TokenType.COMMA or \
                       (self.peek(2).type == TokenType.ARROW and self.peek(2).value == '|->'):
                        type_parts.append(str(self.advance().value))
                        break
                type_parts.append(str(self.advance().value))
            type_annotation = ''.join(type_parts).strip()
        
        parameters = []
        
        # Check for comma before parameters
        if self.match(TokenType.COMMA):
            self.advance()
            self.skip_newlines()
        
        # Parse parameters
        while self.match_identifier():
            parameters.append(self.advance().value)
            if self.match(TokenType.COMMA):
                self.advance()
                self.skip_newlines()
        
        # Expect |-> arrow
        if self.match(TokenType.ARROW):
            self.advance()
            self.skip_newlines()
        
        body = self.parse_expression()
        
        return LambdaExpression(parameters=parameters, body=body, type_annotation=type_annotation)


# ============================================================================
# Code Generator
# ============================================================================

class CodeGenerator:
    """Generates Python code from Befehl AST."""
    
    def __init__(self):
        self.indent_level = 0
        self.indent_str = "    "
        self.imports: set = set()
        
    def indent(self) -> str:
        return self.indent_str * self.indent_level
    
    def generate(self, program: Program) -> str:
        """Generate Python code from the program."""
        lines = []
        lines.append("# Generated Python code from Befehl language")
        lines.append("")
        
        # First pass: collect imports
        for stmt in program.statements:
            if isinstance(stmt, BefehlStatement):
                self._collect_imports(stmt.statement)
        
        # Generate imports
        if self.imports:
            for imp in sorted(self.imports):
                lines.append(imp)
            lines.append("")
        
        # Generate statements
        for stmt in program.statements:
            code = self.generate_statement(stmt)
            if code:
                lines.append(code)
        
        return '\n'.join(lines)
    
    def _collect_imports(self, stmt: ASTNode):
        """Collect import statements."""
        if isinstance(stmt, ImportStatement):
            if not stmt.use_import_func:
                as_name = stmt.as_name
                if as_name and self._is_python_keyword(as_name):
                    as_name = None
                
                if stmt.names:
                    names_str = ', '.join(stmt.names)
                    self.imports.add(f"from {stmt.module} import {names_str}")
                elif as_name:
                    self.imports.add(f"import {stmt.module} as {as_name}")
                else:
                    self.imports.add(f"import {stmt.module}")
        elif isinstance(stmt, CapableCheck):
            self._collect_imports(stmt.statement)
    
    def generate_statement(self, stmt: ASTNode) -> str:
        """Generate Python code for a statement."""
        if isinstance(stmt, BefehlStatement):
            return self.generate_statement(stmt.statement)
        
        if isinstance(stmt, VariableDeclaration):
            return self.generate_variable_declaration(stmt)
        
        if isinstance(stmt, Assignment):
            return self.generate_assignment(stmt)
        
        if isinstance(stmt, FunctionDefinition):
            return self.generate_function_definition(stmt)
        
        if isinstance(stmt, ClassDefinition):
            return self.generate_class_definition(stmt)
        
        if isinstance(stmt, IfStatement):
            return self.generate_if_statement(stmt)
        
        if isinstance(stmt, WhileStatement):
            return self.generate_while_statement(stmt)
        
        if isinstance(stmt, ForEachStatement):
            return self.generate_foreach_statement(stmt)
        
        if isinstance(stmt, PrintStatement):
            return self.generate_print_statement(stmt)
        
        if isinstance(stmt, ReturnStatement):
            return self.generate_return_statement(stmt)
        
        if isinstance(stmt, ImportStatement):
            return self.generate_import_statement(stmt)
        
        if isinstance(stmt, ExpressionStatement):
            return f"{self.indent()}{self.generate_expression(stmt.expression)}"
        
        if isinstance(stmt, CapableCheck):
            return self.generate_capable_check(stmt)
        
        if isinstance(stmt, PythonBlock):
            return self.generate_python_block(stmt)
        
        return ""
    
    def generate_variable_declaration(self, stmt: VariableDeclaration) -> str:
        """Generate variable declaration."""
        name = self.sanitize_identifier(stmt.name)
        value = self.generate_expression(stmt.value)
        return f"{self.indent()}{name} = {value}"
    
    def generate_assignment(self, stmt: Assignment) -> str:
        """Generate assignment."""
        targets = ', '.join(self.generate_expression(t) for t in stmt.targets)
        value = self.generate_expression(stmt.value)
        return f"{self.indent()}{targets} = {value}"
    
    def generate_function_definition(self, stmt: FunctionDefinition) -> str:
        """Generate function definition."""
        name = self.sanitize_identifier(stmt.name)
        params = ', '.join(self.sanitize_identifier(p[0]) for p in stmt.parameters)
        
        lines = [f"{self.indent()}def {name}({params}):"]
        self.indent_level += 1
        
        if stmt.body:
            for body_stmt in stmt.body:
                code = self.generate_statement(body_stmt)
                if code:
                    lines.append(code)
        else:
            lines.append(f"{self.indent()}pass")
        
        self.indent_level -= 1
        return '\n'.join(lines)
    
    def generate_class_definition(self, stmt: ClassDefinition) -> str:
        """Generate class definition."""
        name = self.sanitize_identifier(stmt.name)
        
        if stmt.base_class:
            lines = [f"{self.indent()}class {name}({self.sanitize_identifier(stmt.base_class)}):"]
        else:
            lines = [f"{self.indent()}class {name}:"]
        
        self.indent_level += 1
        
        if stmt.methods:
            for method in stmt.methods:
                code = self.generate_method_definition(method)
                lines.append(code)
                lines.append("")
        else:
            lines.append(f"{self.indent()}pass")
        
        self.indent_level -= 1
        return '\n'.join(lines)
    
    def generate_method_definition(self, stmt: MethodDefinition) -> str:
        """Generate method definition."""
        name = self.sanitize_identifier(stmt.name)
        params = ', '.join(self.sanitize_identifier(p[0]) for p in stmt.parameters)
        
        lines = [f"{self.indent()}def {name}({params}):"]
        self.indent_level += 1
        
        if stmt.body:
            for body_stmt in stmt.body:
                code = self.generate_statement(body_stmt)
                if code:
                    lines.append(code)
        else:
            lines.append(f"{self.indent()}pass")
        
        self.indent_level -= 1
        return '\n'.join(lines)
    
    def generate_if_statement(self, stmt: IfStatement) -> str:
        """Generate if/elif/else statement."""
        lines = []
        
        condition = self.generate_expression(stmt.condition)
        lines.append(f"{self.indent()}if {condition}:")
        
        self.indent_level += 1
        if stmt.then_body:
            for body_stmt in stmt.then_body:
                code = self.generate_statement(body_stmt)
                if code:
                    lines.append(code)
        else:
            lines.append(f"{self.indent()}pass")
        self.indent_level -= 1
        
        for elif_cond, elif_body in stmt.elif_clauses:
            condition = self.generate_expression(elif_cond)
            lines.append(f"{self.indent()}elif {condition}:")
            
            self.indent_level += 1
            if elif_body:
                for body_stmt in elif_body:
                    code = self.generate_statement(body_stmt)
                    if code:
                        lines.append(code)
            else:
                lines.append(f"{self.indent()}pass")
            self.indent_level -= 1
        
        if stmt.else_body:
            lines.append(f"{self.indent()}else:")
            
            self.indent_level += 1
            for body_stmt in stmt.else_body:
                code = self.generate_statement(body_stmt)
                if code:
                    lines.append(code)
            self.indent_level -= 1
        
        return '\n'.join(lines)
    
    def generate_while_statement(self, stmt: WhileStatement) -> str:
        """Generate while loop."""
        lines = []
        
        condition = self.generate_expression(stmt.condition)
        lines.append(f"{self.indent()}while {condition}:")
        
        self.indent_level += 1
        if stmt.body:
            for body_stmt in stmt.body:
                code = self.generate_statement(body_stmt)
                if code:
                    lines.append(code)
        else:
            lines.append(f"{self.indent()}pass")
        self.indent_level -= 1
        
        return '\n'.join(lines)
    
    def generate_foreach_statement(self, stmt: ForEachStatement) -> str:
        """Generate foreach loop."""
        lines = []
        
        if stmt.variable and stmt.iterable:
            iterable = self.generate_expression(stmt.iterable)
            lines.append(f"{self.indent()}for {stmt.variable} in {iterable}:")
        else:
            lines.append(f"{self.indent()}for _ in []:")
        
        self.indent_level += 1
        if stmt.body:
            for body_stmt in stmt.body:
                code = self.generate_statement(body_stmt)
                if code:
                    lines.append(code)
        else:
            lines.append(f"{self.indent()}pass")
        self.indent_level -= 1
        
        return '\n'.join(lines)
    
    def generate_print_statement(self, stmt: PrintStatement) -> str:
        """Generate print statement."""
        value = self.generate_expression(stmt.value)
        return f"{self.indent()}print({value})"
    
    def generate_return_statement(self, stmt: ReturnStatement) -> str:
        """Generate return statement."""
        if stmt.value:
            value = self.generate_expression(stmt.value)
            return f"{self.indent()}return {value}"
        return f"{self.indent()}return"
    
    def generate_import_statement(self, stmt: ImportStatement) -> str:
        """Generate import statement."""
        if stmt.use_import_func:
            return f"{self.indent()}__import__('{stmt.module}')"
        
        # Handle the case where as_name is a Python keyword
        as_name = stmt.as_name
        if as_name and self._is_python_keyword(as_name):
            # Use the module name instead
            as_name = None
        
        if as_name:
            self.imports.add(f"import {stmt.module} as {as_name}")
        return ""
    
    def _is_python_keyword(self, name: str) -> bool:
        """Check if a name is a Python keyword."""
        import keyword
        return keyword.iskeyword(name)
    
    def generate_capable_check(self, stmt: CapableCheck) -> str:
        """Generate capable check (try/except)."""
        lines = []
        lines.append(f"{self.indent()}try:")
        self.indent_level += 1
        
        inner = self.generate_statement(stmt.statement)
        if inner:
            lines.append(inner)
        else:
            lines.append(f"{self.indent()}pass")
        
        self.indent_level -= 1
        lines.append(f"{self.indent()}except Exception:")
        self.indent_level += 1
        lines.append(f"{self.indent()}pass")
        self.indent_level -= 1
        
        return '\n'.join(lines)
    
    def generate_python_block(self, stmt: PythonBlock) -> str:
        """Generate Python block."""
        lines = stmt.code.split('\n')
        return '\n'.join(f"{self.indent()}{line}" for line in lines)
    
    def generate_expression(self, expr: ASTNode) -> str:
        """Generate Python expression."""
        if isinstance(expr, NumberLiteral):
            return str(expr.value)
        
        if isinstance(expr, StringLiteral):
            return repr(expr.value)
        
        if isinstance(expr, FString):
            return f"f{repr(expr.template)}"
        
        if isinstance(expr, Identifier):
            return self.sanitize_identifier(expr.name)
        
        if isinstance(expr, BinaryOp):
            left = self.generate_expression(expr.left)
            right = self.generate_expression(expr.right)
            return f"({left} {expr.op} {right})"
        
        if isinstance(expr, UnaryOp):
            operand = self.generate_expression(expr.operand)
            return f"({expr.op} {operand})"
        
        if isinstance(expr, FunctionCall):
            func = self.generate_expression(expr.function)
            args = ', '.join(self.generate_expression(a) for a in expr.arguments)
            return f"{func}({args})"
        
        if isinstance(expr, MemberAccess):
            obj = self.generate_expression(expr.object)
            return f"{obj}.{expr.member}"
        
        if isinstance(expr, IndexAccess):
            obj = self.generate_expression(expr.object)
            idx = self.generate_expression(expr.index)
            return f"{obj}[{idx}]"
        
        if isinstance(expr, ListLiteral):
            elements = ', '.join(self.generate_expression(e) for e in expr.elements)
            return f"[{elements}]"
        
        if isinstance(expr, DictLiteral):
            pairs = ', '.join(
                f"{self.generate_expression(k)}: {self.generate_expression(v)}"
                for k, v in expr.pairs
            )
            return f"{{{pairs}}}"
        
        if isinstance(expr, SetLiteral):
            elements = ', '.join(self.generate_expression(e) for e in expr.elements)
            return f"{{{elements}}}"
        
        if isinstance(expr, TupleLiteral):
            if len(expr.elements) == 0:
                return "()"
            elements = ', '.join(self.generate_expression(e) for e in expr.elements)
            if len(expr.elements) == 1:
                return f"({elements},)"
            return f"({elements})"
        
        if isinstance(expr, LambdaExpression):
            params = ', '.join(expr.parameters)
            body = self.generate_expression(expr.body)
            return f"lambda {params}: {body}"
        
        return ""
    
    def sanitize_identifier(self, name: str) -> str:
        """Sanitize identifier name for Python."""
        if name.startswith('(') and name.endswith(')'):
            inner = name[1:-1]
            return f"_special_{inner}"
        return name


# ============================================================================
# Interpreter
# ============================================================================

class ReturnException(Exception):
    """Exception used for return statements."""
    def __init__(self, value=None):
        self.value = value


class BefehlInterpreter:
    """Interprets Befehl code directly."""
    
    def __init__(self):
        self.globals: dict = {'__builtins__': __builtins__}
        self.locals_stack: List[dict] = [{}]
        
    @property
    def current_locals(self) -> dict:
        return self.locals_stack[-1]
    
    def get_variable(self, name: str) -> Any:
        if name in self.current_locals:
            return self.current_locals[name]
        if name in self.globals:
            return self.globals[name]
        raise NameError(f"Variable '{name}' is not defined")
    
    def set_variable(self, name: str, value: Any):
        self.current_locals[name] = value
    
    def interpret(self, program: Program):
        for stmt in program.statements:
            self.execute_statement(stmt)
    
    def execute_statement(self, stmt: ASTNode):
        if isinstance(stmt, BefehlStatement):
            self.execute_statement(stmt.statement)
        elif isinstance(stmt, VariableDeclaration):
            value = self.evaluate(stmt.value)
            name = self.sanitize_identifier(stmt.name)
            self.set_variable(name, value)
        elif isinstance(stmt, Assignment):
            value = self.evaluate(stmt.value)
            if len(stmt.targets) == 1:
                target = stmt.targets[0]
                if isinstance(target, Identifier):
                    self.set_variable(self.sanitize_identifier(target.name), value)
                elif isinstance(target, MemberAccess):
                    obj = self.evaluate(target.object)
                    setattr(obj, target.member, value)
                elif isinstance(target, IndexAccess):
                    obj = self.evaluate(target.object)
                    idx = self.evaluate(target.index)
                    obj[idx] = value
            else:
                # Multiple targets
                if hasattr(value, '__iter__'):
                    values = list(value)
                    for i, target in enumerate(stmt.targets):
                        if i < len(values):
                            if isinstance(target, Identifier):
                                self.set_variable(self.sanitize_identifier(target.name), values[i])
                            elif isinstance(target, MemberAccess):
                                obj = self.evaluate(target.object)
                                setattr(obj, target.member, values[i])
        elif isinstance(stmt, FunctionDefinition):
            def make_function(func_def):
                def func(*args):
                    new_locals = {}
                    for i, (param_name, _) in enumerate(func_def.parameters):
                        if i < len(args):
                            new_locals[param_name] = args[i]
                    self.locals_stack.append(new_locals)
                    try:
                        for body_stmt in func_def.body:
                            self.execute_statement(body_stmt)
                    except ReturnException as e:
                        return e.value
                    finally:
                        self.locals_stack.pop()
                    return None
                return func
            self.set_variable(stmt.name, make_function(stmt))
        elif isinstance(stmt, ClassDefinition):
            methods = {}
            for method in stmt.methods:
                def make_method(m):
                    def method_func(*args):
                        new_locals = {}
                        for i, (param_name, _) in enumerate(m.parameters):
                            if i < len(args):
                                new_locals[param_name] = args[i]
                        self.locals_stack.append(new_locals)
                        try:
                            for body_stmt in m.body:
                                self.execute_statement(body_stmt)
                        except ReturnException as e:
                            return e.value
                        finally:
                            self.locals_stack.pop()
                        return None
                    return method_func
                methods[method.name] = make_method(method)
            
            cls = type(stmt.name, (object,), methods)
            self.set_variable(stmt.name, cls)
        elif isinstance(stmt, IfStatement):
            if self.evaluate(stmt.condition):
                for body_stmt in stmt.then_body:
                    self.execute_statement(body_stmt)
            else:
                executed = False
                for elif_cond, elif_body in stmt.elif_clauses:
                    if self.evaluate(elif_cond):
                        for body_stmt in elif_body:
                            self.execute_statement(body_stmt)
                        executed = True
                        break
                if not executed and stmt.else_body:
                    for body_stmt in stmt.else_body:
                        self.execute_statement(body_stmt)
        elif isinstance(stmt, WhileStatement):
            while self.evaluate(stmt.condition):
                for body_stmt in stmt.body:
                    self.execute_statement(body_stmt)
        elif isinstance(stmt, ForEachStatement):
            if stmt.iterable:
                iterable = self.evaluate(stmt.iterable)
                for element in iterable:
                    if stmt.variable:
                        self.set_variable(stmt.variable, element)
                    for body_stmt in stmt.body:
                        self.execute_statement(body_stmt)
        elif isinstance(stmt, PrintStatement):
            value = self.evaluate(stmt.value)
            print(value)
        elif isinstance(stmt, ReturnStatement):
            value = self.evaluate(stmt.value) if stmt.value else None
            raise ReturnException(value)
        elif isinstance(stmt, ImportStatement):
            if stmt.use_import_func:
                module = __import__(stmt.module)
                self.set_variable(stmt.module, module)
            elif stmt.names:
                module = __import__(stmt.module, fromlist=stmt.names)
                for name in stmt.names:
                    self.set_variable(name, getattr(module, name))
            else:
                module = __import__(stmt.module)
                if stmt.as_name:
                    self.set_variable(stmt.as_name, module)
                else:
                    self.set_variable(stmt.module, module)
        elif isinstance(stmt, ExpressionStatement):
            self.evaluate(stmt.expression)
        elif isinstance(stmt, PythonBlock):
            exec(stmt.code, self.globals, self.current_locals)
        elif isinstance(stmt, CapableCheck):
            try:
                self.execute_statement(stmt.statement)
            except:
                pass
    
    def evaluate(self, expr: ASTNode) -> Any:
        if isinstance(expr, NumberLiteral):
            return expr.value
        
        if isinstance(expr, StringLiteral):
            return expr.value
        
        if isinstance(expr, FString):
            template = expr.template
            result = template
            for match in re.finditer(r'\{([^}]+)\}', template):
                var_expr = match.group(1)
                try:
                    value = self.get_variable(var_expr)
                    result = result.replace(match.group(0), str(value))
                except NameError:
                    pass
            return result
        
        if isinstance(expr, Identifier):
            return self.get_variable(self.sanitize_identifier(expr.name))
        
        if isinstance(expr, BinaryOp):
            left = self.evaluate(expr.left)
            right = self.evaluate(expr.right)
            
            ops = {
                '+': lambda a, b: a + b,
                '-': lambda a, b: a - b,
                '*': lambda a, b: a * b,
                '/': lambda a, b: a / b,
                '//': lambda a, b: a // b,
                '%': lambda a, b: a % b,
                '==': lambda a, b: a == b,
                '!=': lambda a, b: a != b,
                '<': lambda a, b: a < b,
                '>': lambda a, b: a > b,
                '<=': lambda a, b: a <= b,
                '>=': lambda a, b: a >= b,
                'and': lambda a, b: a and b,
                'or': lambda a, b: a or b,
                'is': lambda a, b: a is b,
                'in': lambda a, b: a in b,
                'not in': lambda a, b: a not in b,
            }
            
            return ops.get(expr.op, lambda a, b: None)(left, right)
        
        if isinstance(expr, UnaryOp):
            operand = self.evaluate(expr.operand)
            if expr.op == '-':
                return -operand
            if expr.op == 'not':
                return not operand
            return operand
        
        if isinstance(expr, FunctionCall):
            func = self.evaluate(expr.function)
            args = [self.evaluate(a) for a in expr.arguments]
            return func(*args)
        
        if isinstance(expr, MemberAccess):
            obj = self.evaluate(expr.object)
            return getattr(obj, expr.member)
        
        if isinstance(expr, IndexAccess):
            obj = self.evaluate(expr.object)
            idx = self.evaluate(expr.index)
            return obj[idx]
        
        if isinstance(expr, ListLiteral):
            return [self.evaluate(e) for e in expr.elements]
        
        if isinstance(expr, DictLiteral):
            return {self.evaluate(k): self.evaluate(v) for k, v in expr.pairs}
        
        if isinstance(expr, SetLiteral):
            return {self.evaluate(e) for e in expr.elements}
        
        if isinstance(expr, TupleLiteral):
            return tuple(self.evaluate(e) for e in expr.elements)
        
        if isinstance(expr, LambdaExpression):
            def make_lambda(lam):
                def lambda_func(*args):
                    new_locals = {}
                    for i, param in enumerate(lam.parameters):
                        if i < len(args):
                            new_locals[param] = args[i]
                    self.locals_stack.append(new_locals)
                    try:
                        return self.evaluate(lam.body)
                    finally:
                        self.locals_stack.pop()
                return lambda_func
            return make_lambda(expr)
        
        return None
    
    def sanitize_identifier(self, name: str) -> str:
        if name.startswith('(') and name.endswith(')'):
            inner = name[1:-1]
            return f"_special_{inner}"
        return name


# ============================================================================
# Compiler Class
# ============================================================================

class BefehlCompiler:
    """Main compiler class."""
    
    def __init__(self, source: str):
        self.source = source
        self.tokens: List[Token] = []
        self.ast: Optional[Program] = None
        self.python_code: Optional[str] = None
    
    def compile(self) -> str:
        """Compile Befehl source to Python code."""
        lexer = Lexer(self.source)
        self.tokens = lexer.tokenize()
        
        parser = Parser(self.tokens)
        self.ast = parser.parse()
        
        generator = CodeGenerator()
        self.python_code = generator.generate(self.ast)
        
        return self.python_code
    
    def execute(self):
        """Compile and execute the Befehl code."""
        python_code = self.compile()
        exec(python_code, {})
    
    def interpret(self):
        """Interpret the Befehl code directly."""
        lexer = Lexer(self.source)
        self.tokens = lexer.tokenize()
        
        parser = Parser(self.tokens)
        self.ast = parser.parse()
        
        interpreter = BefehlInterpreter()
        interpreter.interpret(self.ast)


# ============================================================================
# Main Entry Point
# ============================================================================

def compile_befehl(source: str, output_file: str = None) -> str:
    """Compile Befehl source code to Python."""
    compiler = BefehlCompiler(source)
    python_code = compiler.compile()
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(python_code)
    
    return python_code


def run_befehl(source: str, use_interpreter: bool = False):
    """Run Befehl source code."""
    compiler = BefehlCompiler(source)
    
    if use_interpreter:
        compiler.interpret()
    else:
        compiler.execute()


if __name__ == "__main__":
    example_code = '''
    Befehl: a := 1
    print(a)
    
    Befehl: let function with name `add` <- x, y {
        return x + y
    }
    
    print(add(1, 2))
    '''
    
    print("Compiling Befehl code...")
    python_code = compile_befehl(example_code)
    print("Generated Python code:")
    print(python_code)
    print("\nExecuting...")
    run_befehl(example_code)
