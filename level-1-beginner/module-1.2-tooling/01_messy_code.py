# 01 — Messy code (intentionally!)
#
# This file is DELIBERATELY badly formatted and lightly broken to demonstrate
# what ruff and black do.
#
# Try, in order:
#   ruff check 01_messy_code.py            # see the warnings
#   ruff check --fix 01_messy_code.py      # auto-fix what's safe
#   ruff format 01_messy_code.py           # reformat (or: black 01_messy_code.py)
#
# Then re-open the file and see how it changed.
# (Use `git diff` if you want to inspect line-by-line — that's a great habit.)

import sys, os                             # multiple-imports-on-one-line (E401)
import json                                # unused import (F401)


def calc(x,y):                             # missing-whitespace-after-comma (E231)
    a   =   x +y                            # multiple-spaces (E222), missing-whitespace (E225)
    b='hello'                              # missing-whitespace-around-operator (E225)
    return a*2


def    bad_function( arg1,arg2 ):          # bad spacing all over (E211, E201, E202)
    print( "hello" )                        # whitespace inside parens
    if arg1==None:                          # comparison-to-none (E711) — should be `is None`
        return 0
    result=[i*2 for i in range(arg1) if i%2==0]
    return result


# Variable name shadowing a built-in (A001) — bad practice
list = [1,2,3]


# Function name CamelCase — should be snake_case (N802)
def DoStuff():
    pass


if __name__=="__main__":
    x = calc(5,10)
    y = bad_function(20,'world')
    print( x,y )
