#!/usr/bin/python

import sys
logs = sys.stderr

from tree import *
import gflags as flags
FLAGS=flags.FLAGS

from collections import defaultdict, OrderedDict
from math import log, exp
from heapdict import heapdict

def process_grammar_line(line):
    line = line.strip().split("->")
    if len(line) == 1:
        return line[0], None
    X, rhs = line
    gamma,p = rhs.strip().split("#")
    X = X.strip()
    gamma = tuple(gamma.strip().split(" "))
    p = log(float(p))
    return X, gamma, p

def read_grammar(lines):
    TOP = lines[0].strip()
    nonterminals,terminals = set(), set()
    grammar_table = defaultdict(lambda: defaultdict(set))
    pos_table = defaultdict(lambda: defaultdict(float))
    first_table = defaultdict(set)

    # To get nonterminals
    for line in lines[1:]:
        X, gamma, p = process_grammar_line(line)
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

    # The first_table is integrated into the grammar table
    for line in lines[1:]:
        X, gamma, p = process_grammar_line(line)
        for a in first_table[gamma[0]]:
            grammar_table[X][a].add((gamma, p))
                    
    return TOP, nonterminals, grammar_table, pos_table

def finished(state):
    return state[1] is None or state[2] == len(state[1])

def next_element_of(state):
    return state[1][state[2]]

def print_chart(chart):
    for i,x in enumerate(chart):
        print "STATE {}: {}".format(i, len(x))
        for j,y in enumerate(sorted(x.keys())):
            if finished(y):
                print "*",
            else:
                print " ",
            print j, y, "\t", x[y]
        print "-"*80
    print
    
def INIT(words):
    chart, backptrs, unfinished, completed = [], [], [], []
    for x in xrange(len(words)+1):
        chart.append(OrderedDict())
        backptrs.append(defaultdict(list))
        unfinished.append(defaultdict(set))
        completed.append(heapdict())
    return chart, backptrs, unfinished, completed

def EARLEY_PARSE(words, grammar):
    TOP, nonterminals, grammar_table, pos_table = grammar
    chart, backptrs, unfinished, completed = INIT(words)
    chart[0][(TOP, (TOP,), 0, 0)] = 0.0

    for k in xrange(len(words)):
        for i,state in enumerate(chart[k]):
            #print k, i, state, chart[k][state]
            if not finished(state):
                if next_element_of(state) in nonterminals:
                    chart, backptrs, unfinished = PREDICTOR(chart, backptrs, unfinished,
                                                            state, k, grammar_table, words[k])
                else:
                    chart, unfinished = SCANNER(chart, unfinished,
                                                state, k, pos_table, words)
            else:
                chart, backptrs, unfinished, completed = COMPLETER(chart, backptrs, unfinished, completed, state, k)
            if i == len(chart[k])-1 and completed[k]:
                (A,j), (prob, completed_state, state_backptrs) = completed[k].popitem()
                chart[k][completed_state] = -prob
                backptrs[k][completed_state] = state_backptrs
    
    for i,state in enumerate(chart[k+1]):
        if finished(state):
            chart, backptrs, unfinished, completed = COMPLETER(chart, backptrs, unfinished, completed, state, k+1)
        if i == len(chart[k+1])-1 and completed[k+1]:
            (A,j), (prob, completed_state, state_backptrs) = completed[k+1].popitem()
            chart[k+1][completed_state] = -prob
            backptrs[k+1][completed_state] = state_backptrs
            
    return chart, backptrs

def PREDICTOR(chart, backptrs, unfinished,
              state, k, grammar_table, word):
    X = next_element_of(state)
    for gamma,p in grammar_table[X][word]:
        new_state = (X, gamma, 0, k)
        if (X, gamma, 0, k) not in chart[k]:
            chart[k][new_state] = p
            unfinished[k][next_element_of(new_state)].add(new_state)
    return chart, backptrs, unfinished

def SCANNER(chart, unfinished,
            state, k, pos_table, words):
    X, gamma, i, j = state
    a = next_element_of(state)
    A = state[0]
    if A in pos_table[words[k]]:
        progressed_state = (X, gamma, i+1, j)
        chart[k+1][progressed_state] = pos_table[words[k]][A]
        if not finished(progressed_state):
            unfinished[k+1][next_element_of(progressed_state)].add(progressed_state)
    return chart, unfinished

def COMPLETER(chart, backptrs, unfinished, completed, completed_state, k):
    B, gamma, i, x = completed_state

    for incomplete_state in unfinished[x][B]:
        A, gamma2, i2, j = incomplete_state
        p1 = chart[k][completed_state]
        p2 = chart[x][incomplete_state]
        prob = p1+p2
        progressed_state = (A, gamma2, i2+1, j)

        if finished(progressed_state):
            if progressed_state in chart[k]:
                continue
            if (A,j) in completed[k]:
                if prob > -completed[k][(A,j)][0]:
                    new_backptrs = [prev_backptr for prev_backptr in backptrs[x][incomplete_state]]
                    new_backptrs.append((completed_state, k))
                    completed[k][(A,j)] = (-prob, progressed_state, new_backptrs)
            else:
                new_backptrs = [prev_backptr for prev_backptr in backptrs[x][incomplete_state]]
                new_backptrs.append((completed_state, k))
                completed[k][(A,j)] = (-prob, progressed_state, new_backptrs)
        else:
            if progressed_state not in chart[k] or prob > chart[k][progressed_state]:
                chart[k][progressed_state] = prob
                new_backptrs = [prev_backptr for prev_backptr in backptrs[x][incomplete_state]]
                new_backptrs.append((completed_state, k))
                backptrs[k][progressed_state] = new_backptrs
                unfinished[k][next_element_of(progressed_state)].add(progressed_state)
    return chart, backptrs, unfinished, completed


def BACKTRACK(chart, backptrs, state, k, nonterminals):
    root = Tree(label=state[0], span=(0,0), wrd=None, subs=[])
    tovisit = [(state, root, k)]
    while tovisit:
        state, tree, k = tovisit.pop()
        back_list = backptrs[k][state]

        #print state, back_list
        
        for back in back_list:
            back_state, back_k = back
            (X, alpha, k, j) = back_state
            subtree = Tree(label=X, span=(0,0), wrd=None, subs=[])
            tree.subs.append(subtree)
            terminal = True
            for a in alpha:
                if a in nonterminals:
                    terminal = False
                    break
            if terminal:
                subtree.word = " ".join(back_state[1])
            else:
                tovisit.append((back_state, subtree, back_k))
    result = Tree.parse(str(root))
    return result
        
if __name__ == "__main__":
    sentences = [x.strip() for x in sys.stdin.readlines()]
    grammar = read_grammar(open(sys.argv[1], 'r').readlines())
    TOP, nonterminals, grammar_table, pos_table = grammar

    for sentence in sentences:
        words = sentence.split()
        chart,backptrs = EARLEY_PARSE(words, grammar)
        #print "CHART:"
        #print_chart(chart)
        
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
        tree = BACKTRACK(chart, backptrs, best_state, len(chart)-1, nonterminals)
        print tree, exp(best_prob)
