"""
Exercise 4 — A small expression calculator with custom exceptions

Supports + - * / and parentheses on integers and floats.
Demonstrates: tokenizing, recursive-descent parsing, custom exceptions.

Usage:
    python exercises/04_calculator.py "2 + 3 * (4 - 1)"
"""

import sys
import re


class CalculatorError(Exception):
    """Base class for calculator errors."""


class TokenizeError(CalculatorError):
    pass


class ParseError(CalculatorError):
    pass


class EvalError(CalculatorError):
    pass


TOKEN_RE = re.compile(r"\s*(?:(\d+\.\d+|\d+)|([+\-*/()]))")


def tokenize(text: str) -> list[str]:
    pos = 0
    tokens: list[str] = []
    while pos < len(text):
        m = TOKEN_RE.match(text, pos)
        if not m:
            raise TokenizeError(f"unexpected character at position {pos}: {text[pos]!r}")
        if m.group(0).strip() == "":
            break
        tokens.append(m.group(1) or m.group(2))
        pos = m.end()
    return tokens


class Parser:
    """Grammar:
        expr   := term (('+' | '-') term)*
        term   := factor (('*' | '/') factor)*
        factor := NUMBER | '(' expr ')' | '-' factor
    """

    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.i = 0

    def peek(self) -> str | None:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def take(self) -> str:
        tok = self.peek()
        if tok is None:
            raise ParseError("unexpected end of expression")
        self.i += 1
        return tok

    def parse(self) -> float:
        result = self.expr()
        if self.peek() is not None:
            raise ParseError(f"unexpected token: {self.peek()!r}")
        return result

    def expr(self) -> float:
        value = self.term()
        while self.peek() in ("+", "-"):
            op = self.take()
            rhs = self.term()
            value = value + rhs if op == "+" else value - rhs
        return value

    def term(self) -> float:
        value = self.factor()
        while self.peek() in ("*", "/"):
            op = self.take()
            rhs = self.factor()
            if op == "*":
                value = value * rhs
            else:
                if rhs == 0:
                    raise EvalError("division by zero")
                value = value / rhs
        return value

    def factor(self) -> float:
        tok = self.take()
        if tok == "(":
            value = self.expr()
            if self.take() != ")":
                raise ParseError("missing closing parenthesis")
            return value
        if tok == "-":
            return -self.factor()
        try:
            return float(tok)
        except ValueError as e:
            raise ParseError(f"expected a number, got {tok!r}") from e


def evaluate(expression: str) -> float:
    tokens = tokenize(expression)
    if not tokens:
        raise ParseError("empty expression")
    return Parser(tokens).parse()


def main():
    if len(sys.argv) < 2:
        sys.exit('usage: python 04_calculator.py "2 + 3 * (4 - 1)"')

    expression = " ".join(sys.argv[1:])
    try:
        result = evaluate(expression)
    except CalculatorError as e:
        sys.exit(f"{type(e).__name__}: {e}")

    if result == int(result):
        result = int(result)
    print(result)


if __name__ == "__main__":
    main()
