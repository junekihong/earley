#!/usr/bin/python

import sys
logs = sys.stderr

from tree import *
import gflags as flags
FLAGS=flags.FLAGS

from collections import defaultdict, OrderedDict
from heapdict import heapdict
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

    """
    for X in grammar_table:
        for a in grammar_table[X]:
            #print grammar_table[X][a]
            grammar_table[X][a] = sorted(grammar_table[X][a], key=lambda item: item[1], reverse=True)
            #print grammar_table[X][a]
    #exit()
    """

    return TOP, nonterminals, grammar_table, pos_table

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
    chart, backptrs, unfinished = [], [], []
    for x in xrange(len(words)+1):
        chart.append(OrderedDict())
        backptrs.append(defaultdict(list))
        unfinished.append(defaultdict(set))
    return chart, backptrs, unfinished

def EARLEY_PARSE(words, grammar):
    TOP, nonterminals, grammar_table, pos_table = grammar
    chart, backptrs, unfinished = INIT(words)


    initial = ("INITIALIZE", (TOP,), 0, 0)
    chart[0][initial] = 1.0
    queue = heapdict()
    queue[initial] = -chart[0][initial]


    for k in xrange(len(words)):
        while queue:
            #print queue.peekitem()
            state,_ = queue.popitem()

            
            if not finished(state):
                if next_element_of(state) in nonterminals:
                    queue, chart, backptrs, unfinished = PREDICTOR(queue, chart, backptrs, unfinished,
                                                                   state, k, grammar_table, words[k])
                else:
                    queue, chart, unfinished = SCANNER(queue, chart, unfinished,
                                                       state, k, pos_table, words)
            else:
                queue, chart, backptrs, unfinished = COMPLETER(queue, chart, backptrs, unfinished, state, k)
        for state in chart[k+1]:
            queue[state] = chart[k+1][state]

    """
    for k in xrange(len(words)):
        for state in chart[k]:
            if not finished(state):
                if next_element_of(state) in nonterminals:
                    chart, backptrs, unfinished = PREDICTOR(chart, backptrs, unfinished,
                                                            state, k, grammar_table, words[k])
                else:
                    chart, unfinished = SCANNER(chart, unfinished,
                                                state, k, pos_table, words)
            else:
                chart, backptrs, unfinished = COMPLETER(chart, backptrs, unfinished, state, k)
    """
    
    for state in chart[k+1]:
        if finished(state):
            queue, chart, backptrs, unfinished = COMPLETER(queue, chart, backptrs, unfinished, state, k+1)
    return chart, backptrs

def PREDICTOR(queue, chart, backptrs, unfinished,
              state, k, grammar_table, word):
    X = next_element_of(state)
    for gamma,p in grammar_table[X][word]:
        new_state = (X, gamma, 0, k)
        if (X, gamma, 0, k) not in chart[k]:
            chart[k][new_state] = p
            unfinished[k][next_element_of(new_state)].add(new_state)
            queue[new_state] = -p
        elif p > chart[k][(X, gamma, 0, k)]:
            del chart[k][new_state]
            chart[k][new_state] = p
            queue[new_state] = -p
    return queue, chart, backptrs, unfinished

def SCANNER(queue, chart, unfinished,
            state, k, pos_table, words):
    X, gamma, i, j = state
    a = next_element_of(state)
    A = state[0]
    if pos_table[words[k]][A] != 0:
        progressed_state = (X, gamma, i+1, j)
        chart[k+1][progressed_state] = pos_table[words[k]][A]
        #queue[progressed_state] = -pos_table[words[k]][A]
        
        if not finished(progressed_state):
            unfinished[k+1][next_element_of(progressed_state)].add(progressed_state)
    return queue, chart, unfinished

def COMPLETER(queue, chart, backptrs, unfinished, completed_state, k):
    B, gamma, i, x = completed_state

    # Sort unfinished states looking for a B in order by weight
    #if len(unfinished[x][B]) > 5:
    #candidates = sorted([(chart[x][incomplete_state], incomplete_state)
    #                     for incomplete_state in unfinished[x][B]],
    #                    reverse=True)
    #else:
    #    candidates = [(chart[x][incomplete_state], incomplete_state)
    #                  for incomplete_state in unfinished[x][B]]
    candidates = unfinished[x][B]
    for incomplete_state in candidates:
        A, gamma2, i2, j = incomplete_state

        #print completed_state
        #print k, chart[k]
        
        p1 = chart[k][completed_state]
        p2 = chart[x][incomplete_state]
        progressed_state = (A, gamma2, i2+1, j)
        
        if progressed_state not in chart[k]:
            chart[k][progressed_state] = p1*p2
            queue[progressed_state] = -p1*p2
            
            for prev_backptr in backptrs[x][incomplete_state]:
                backptrs[k][progressed_state].append(prev_backptr)
            backptrs[k][progressed_state].append((completed_state, k))
            if not finished(progressed_state):
                unfinished[k][next_element_of(progressed_state)].add(progressed_state)
                
        elif p1*p2 > chart[k][progressed_state]:
            del chart[k][progressed_state]
            backptrs[k][progressed_state] = []
            chart[k][progressed_state] = p1*p2
            queue[progressed_state] = -p1*p2
            
            for prev_backptr in backptrs[x][incomplete_state]:
                backptrs[k][progressed_state].append(prev_backptr)
            backptrs[k][progressed_state].append((completed_state, k))
            
    return queue, chart, backptrs, unfinished


def BACKTRACK(chart, backptrs, state, k, visited, nonterminals):
    root = Tree(label=state[0], span=(0,0), wrd=None, subs=[])
    tovisit = [(state, root, k)]
    while tovisit:
        state, tree, k = tovisit.pop()
        back_list = backptrs[k][state]
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
        visited = set([best_state])
        tree = BACKTRACK(chart, backptrs, best_state, len(chart)-1, visited, nonterminals)
        print tree, best_prob
