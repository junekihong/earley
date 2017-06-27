#!/usr/bin/python

import sys
logs = sys.stderr

from tree import *
import gflags as flags
FLAGS=flags.FLAGS

from collections import defaultdict, OrderedDict
from pprint import pprint

def process_grammar_line(line):
    line = line.strip().split("->")
    if len(line) == 1:
        return line[0], None
    X, rhs = line
    gamma,p = rhs.strip().split("#")
    X = X.strip()
    gamma = tuple(gamma.strip().split(" "))
    #p = float(p)
    return X, gamma

def read_grammar(lines):
    nonterminals = set()
    grammar_table = defaultdict(set)
    pos_table = defaultdict(set)
    vocabulary = set()
    #first_table = defaultdict(set())
    TOP = lines[0]
    for line in lines[1:]:
        X, gamma = process_grammar_line(line)
        grammar_table[X].add(gamma)
        nonterminals.add(X)
    for line in lines[1:]:
        X, gamma = process_grammar_line(line)
        for alpha in gamma:
            if alpha not in nonterminals:
                vocabulary.add(alpha)
                if len(gamma) == 1:
                    pos_table[gamma[0]].add(X)
    return nonterminals, grammar_table, pos_table


def finished(state):
    return state[2] == len(state[1])

def next_element_of(state):
    return state[1][state[2]]

def print_chart(chart):
    for i,x in enumerate(chart):
        print "STATE {}:".format(i)
        for y in x:
            print y, "\t", x[y]
        print "-"*80
    print
def INIT(words):
    S = []
    for x in xrange(len(words)+1):
        S.append(OrderedDict())
    return S

def EARLEY_PARSE(words, grammar):
    nonterminals, grammar_table, pos_table = grammar
    S = INIT(words)
    S[0][("TOP", ("S",), 0, 0)] = None
    for k in xrange(len(words)):
        for state in S[k]:
            if not finished(state):
                if next_element_of(state) in nonterminals:
                    #print state
                    #print_chart(S)
                    S = PREDICTOR(S, state, k, grammar_table)
                else:
                    if next_element_of(state) != words[k]:
                        continue
                    S = SCANNER(S, state, k, pos_table, words)
            else:
                S = COMPLETER(S, state, k)
    for state in S[k+1]:
        S = COMPLETER(S, state, k+1)
    return S

def PREDICTOR(S, state, k, grammar_table):
    X = next_element_of(state)
    for gamma in grammar_table[X]:
        S[k][(X, gamma, 0, k)] = state
    return S

def SCANNER(S, state, k, pos_table, words):
    X, gamma, i, j = state
    a = next_element_of(state)
    A = state[0]
    if A in pos_table[words[k]]:
        S[k+1][(X, gamma, i+1, j)] = state
    return S

def COMPLETER(S, state, k):
    B, gamma, i, x = state
    for state in S[x]:
        A, gamma2, i2, j = state
        if finished(state):
            continue
        if gamma2[i2] == B:
            S[k][(A, gamma2, i2+1,j)] = state
    return S


if __name__ == "__main__":
    sentences = [x.strip() for x in sys.stdin.readlines()]

    TOP = None
    nonterminals, grammar_table, pos_table = read_grammar(open(sys.argv[1], 'r').readlines())

    for sentence in sentences:
        words = sentence.split()
        chart = EARLEY_PARSE(words, (nonterminals, grammar_table, pos_table))
        print "CHART:"
        #print chart
        print_chart(chart)
