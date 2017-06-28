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
    p = float(p)
    return X, gamma, p

def read_grammar(lines):
    TOP = lines[0].strip()
    nonterminals,terminals = set(), set()
    grammar_table = defaultdict(set)
    pos_table = defaultdict(lambda: defaultdict(float))
    first_table = defaultdict(set)

    # To get nonterminals
    for line in lines[1:]:
        X, gamma, p = process_grammar_line(line)
        grammar_table[X].add((gamma, p))
        nonterminals.add(X)

    # To get terminals and to initialize the first_table
    for line in lines[1:]:
        X, gamma, p = process_grammar_line(line)
        for alpha in gamma:
            if alpha not in nonterminals:
                terminals.add(alpha)
                first_table[alpha].add(alpha)
                pos_table[alpha][X] = p
        if gamma[0] not in nonterminals:
            first_table[X].add(gamma[0])

    first_table_updates = -1
    while first_table_updates != 0:
        first_table_updates = 0
        for line in lines[1:]:
            X, gamma, _ = process_grammar_line(line)
            Y = gamma[0]

            for a in first_table[Y]:
                if a not in first_table[X]:
                    first_table[X].add(a)
                    first_table_updates += 1
            #print first_table_updates, X, Y
    return TOP, nonterminals, grammar_table, pos_table, first_table

def finished(state):
    return state[2] == len(state[1])

def next_element_of(state):
    return state[1][state[2]]

def print_chart(chart):
    for i,x in enumerate(chart):
        print "STATE {}:".format(i)
        for y in sorted(x.keys()):
            #if finished(y):
            print y, "\t", x[y]
        print "-"*80
    print
    
def INIT(words):
    chart, backptrs = [], []
    for x in xrange(len(words)+1):
        chart.append(OrderedDict())
        backptrs.append(defaultdict(list))
    return chart, backptrs

def EARLEY_PARSE(words, grammar):
    TOP, nonterminals, grammar_table, pos_table, first_table = grammar
    chart, backptrs = INIT(words)
    chart[0][("INITIALIZE", (TOP,), 0, 0)] = 1.0
    for k in xrange(len(words)):
        for state in chart[k]:
            if not finished(state):
                if next_element_of(state) in nonterminals:
                    chart, backptrs = PREDICTOR(chart, backptrs,
                                                state, k, grammar_table,
                                                first_table, words[k])
                else:
                    if next_element_of(state) != words[k]:
                        continue
                    chart, backptrs = SCANNER(chart, backptrs,
                                              state, k, pos_table, words)
            else:
                chart, backptrs = COMPLETER(chart, backptrs, state, k)
    for state in chart[k+1]:
        chart, backptrs = COMPLETER(chart, backptrs, state, k+1)
    return chart, backptrs

def PREDICTOR(chart, backptrs,
              state, k, grammar_table,
              first_table=None, word=None):
    X = next_element_of(state)
    for gamma,p in grammar_table[X]:
        if (first_table is not None and
            word is not None and
            word not in first_table[gamma[0]]):
            continue
        
        new_state = (X, gamma, 0, k)
        if (X, gamma, 0, k) not in chart[k]:
            chart[k][(X, gamma, 0, k)] = p
            backptrs[k][(X, gamma, 0, k)] = [(state, k)]
        elif p > chart[k][(X, gamma, 0, k)]:
            del chart[k][(X, gamma, 0, k)]
            chart[k][(X, gamma, 0, k)] = p
            backptrs[k][(X, gamma, 0, k)] = [(state, k)]
    return chart, backptrs

def SCANNER(chart, backptrs, state, k, pos_table, words):
    X, gamma, i, j = state
    a = next_element_of(state)
    A = state[0]
    if pos_table[words[k]][A] != 0:
        chart[k+1][(X, gamma, i+1, j)] = pos_table[words[k]][A]
        backptrs[k+1][(X, gamma, i+1, j)] = [(state, k)]
    return chart, backptrs

def COMPLETER(chart, backptrs, completed_state, k):
    B, gamma, i, x = completed_state
    for incomplete_state in chart[x]:
        A, gamma2, i2, j = incomplete_state
        if finished(incomplete_state) or not finished(completed_state):
            continue
        if gamma2[i2] == B:
            p1 = chart[k][completed_state]
            p2 = chart[x][incomplete_state]
            progressed_state = (A, gamma2, i2+1, j)
            if progressed_state not in chart[k]:
                chart[k][progressed_state] = p1*p2
                for prev_backptr in backptrs[x][incomplete_state]:
                    if finished(prev_backptr[0]):
                        backptrs[k][progressed_state].append(prev_backptr)
                backptrs[k][progressed_state].append((completed_state, k))
            elif p1*p2 > chart[k][progressed_state]:
                del chart[k][progressed_state]
                del backptrs[k][progressed_state]
                chart[k][progressed_state] = p1*p2
                for prev_backptr in backptrs[x][incomplete_state]:
                    if finished(prev_backptr[0]):
                        backptrs[k][progressed_state].append(prev_backptr)
                backptrs[k][progressed_state].append((completed_state, k))
    return chart, backptrs


def BACKTRACK(chart, backptrs, state, k, visited, nonterminals):
    root = Tree(label=state[0], span=(0,0), wrd=None, subs=[])
    tovisit = [(state, root, k)]
    while tovisit:
        state, tree, k = tovisit.pop()
        back_list = [back for back in backptrs[k][state] if finished(back[0]) and back[0] not in visited]
        if not finished(state):
            continue
        for back in back_list:
            back_state, back_k = back
            visited.add(back_state)
            subtree = Tree(label=back_state[0], span=(0,0), wrd=None, subs=[])
            tree.subs.append(subtree)
            terminal = True
            for alpha in back_state[1]:
                if alpha in nonterminals:
                    terminal = False
                    break
            if terminal:
                subtree.word = " ".join(back_state[1])
            else:
                tovisit.append((back_state, subtree, back_k))
    return root
        
if __name__ == "__main__":
    sentences = [x.strip() for x in sys.stdin.readlines()]
    grammar = read_grammar(open(sys.argv[1], 'r').readlines())
    TOP, nonterminals, grammar_table, pos_table, first_table = grammar

    for sentence in sentences:
        words = sentence.split()
        chart,backptrs = EARLEY_PARSE(words, grammar)
        # print "CHART:"
        # print_chart(chart)
        
        best_state, best_prob = None, None
        for state in chart[len(chart)-1]:
            if not finished(state) or state[0] != TOP:
                continue
            probability = chart[len(chart)-1][state]
            if best_state is None or probability > best_prob:
                best_state = state
                best_prob = probability
        if best_state is None:
            continue
        visited = set([best_state])
        tree = BACKTRACK(chart, backptrs, best_state, len(chart)-1, visited, nonterminals)
        print tree, best_prob
