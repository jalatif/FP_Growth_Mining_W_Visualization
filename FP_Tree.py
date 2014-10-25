__author__ = 'manshu'

import Queue
import itertools
import re
import os
import time
import sys

class TreeNode:
    def __init__(self, name, fq_value):
        self.childs = {}
        self.parent = None
        self.name = name
        self.fq_value = fq_value
        self.graph_node = None

class HeaderTable:
    def __init__(self, item, frequency):
        self.item = item
        self.frequency = frequency
        self.head = None

class LinkedListNode:
    def __init__(self, Node, value):
        self.value = value
        self.node = Node
        self.next = None

def make_fp_tree(transactions, header_table):
    root = TreeNode("{null}", 0);
    if make_graph:
        Groot = U.newVertex(shape="sphere", color="#ffff00", size="1.5", label=str(root.name), fontsize="32")
        root.graph_node = Groot
    for transaction in transactions:
        temp_root = root
        if make_graph: Gtemp_root = Groot
        trans_freq = transaction[0]
        for i in range(0, len(transaction[-1])):
            item = transaction[-1][i]
            if (temp_root.childs.has_key(item)):
                temp_root.childs[item].fq_value += trans_freq
                if make_graph: temp_root.childs[item].graph_node.set(label=str(item) + " " + str(temp_root.childs[item].fq_value))
            else:
                temp_root.childs[item] = TreeNode(item, trans_freq)
                temp_root.childs[item].parent = temp_root
                LNode = header_table[item].head
                new_node = LinkedListNode(temp_root.childs[item], item)
                if LNode:
                    new_node.next = LNode
                    header_table[item].head = new_node
                else:
                    header_table[item].head = new_node
                if make_graph:
                    Gtemp_node = U.newVertex(shape="sphere", color="#ffff00", size="1.5", label=str(item) + " 1", fontsize="32")
                    U.newEdge(Gtemp_root, Gtemp_node, arrow="true", spline="false", stroke="solid", strength="1.0", oriented="true", fontsize="32", fontcolor="#31ef24", color="#ee3333")
                    temp_root.childs[item].graph_node = Gtemp_node
            if make_graph: Gtemp_root = temp_root.childs[item].graph_node
            temp_root = temp_root.childs[item]
    return root

def myprint_fp_tree(tree_root):
    queue = Queue.Queue()
    queue.put(tree_root)
    while not queue.empty():
        node = queue.get()
        myprint(str(node.name) + " " + str(node.fq_value))
        for child in node.childs.keys():
            queue.put(node.childs[child])

def make_ordered_transaction(transactions, flist):
    ordered_transactions = []
    for transaction in transactions:
        ordered_transactions.append([])
        unordered_items = []
        trans_freq = transaction[0]
        ordered_transactions[-1].append(trans_freq)
        ordered_transactions[-1].append([])
        for item in transaction[-1]:
            if item not in flist:
                continue
            else:
                unordered_items.append(item)
        for ordered_item in flist:
            if ordered_item in unordered_items:
                ordered_transactions[-1][-1].append(ordered_item)

    return ordered_transactions

def make_flist(transactions, support):
    header_table = {}
    for transaction in transactions:
        trans_freq = transaction[0]
        for item in transaction[-1]:
            if (item in header_table.keys()):
                header_table[item].frequency += trans_freq
            else:
                header_table[item] = HeaderTable(item, trans_freq)
    del_list = []
    for key in header_table:
        if (header_table[key].frequency < support):
            del_list.append(key)

    for element in del_list:
        del header_table[element]

    return header_table

def checkPatternEnding(tree):
    temp_node = tree
    single = True
    while (temp_node):
        if len(temp_node.childs) > 1:
            single = False
            break
        elif len(temp_node.childs) == 1:
            temp_node = temp_node.childs[temp_node.childs.keys()[0]]
        else:
            break
    return single

def giveCombinations(tree):
    temp_node = tree
    if not temp_node: return None
    itemfreqlist = {}
    if not checkPatternEnding(tree): return None
    while (temp_node):
        if len(temp_node.childs) == 1:
            temp_node = temp_node.childs[temp_node.childs.keys()[0]]
            itemfreqlist[temp_node.name] = temp_node.fq_value
        else:
            break
    itemlist = itemfreqlist.keys()
    itemListCombinations = []
    for i in range(1, len(itemlist) + 1):
        comb = itertools.combinations(itemlist, i)
        for itemcom in comb:
            #itemListCombinations.append([])
            min_support = min([itemfreqlist[item] for item in itemcom])
            patternCombination = []
            for item in itemcom:
                patternCombination.append(item)
            itemListCombinations.append([min_support, patternCombination])
    return itemListCombinations

def myprint(string):
    pass#print string

def mine_fp_tree(transactions, conditional_base = [], min_support = 1, make_graph=False):
    header_table = make_flist(transactions, min_support)
    flist = sorted(header_table.keys(), key=lambda key: header_table[key].frequency, reverse=True)

    for key in flist:
        myprint(str(key) + " " + str(header_table[key].frequency))

    transactions = make_ordered_transaction(transactions, flist)

    for transaction in transactions:
        myprint(transaction)

    tree = make_fp_tree(transactions, header_table)
    myprint_fp_tree(tree)
    myprint("=========================================================")
    myprint("=========================================================")

    if (checkPatternEnding(tree)):
        item_combinations = giveCombinations(tree)
        if item_combinations:
            #print item_combinations
            for item_list in item_combinations:
                PatternCombinations.append([item_list[0], item_list[1] + conditional_base])
        return

    if (len(tree.childs) == 0): return

    for item in header_table:
        string = str(header_table[item].item) + " " + str(header_table[item].frequency)
        temp_node = header_table[item].head
        while (temp_node):
            string += " " + str(temp_node.value) + " " + str(len(temp_node.node.childs))
            temp_node = temp_node.next
        myprint(string)

    conditional_pattern_base = {}
    for item in header_table:
        head = header_table[item].head
        plist = []
        while (head):
            node = head.node.parent
            freq = head.node.fq_value
            name = ""
            plist.append([])
            plist[-1].append([])
            while (node.name != "{null}"):
                name = str(node.name) + " " + name
                plist[-1][-1] = [node.name] + plist[-1][-1]
                node = node.parent
            head = head.next
            if len(plist[-1][-1]) == 0:
                plist.pop()
            else:
                plist[-1] = [freq] + plist[-1]
                name += " : " + str(freq)
                myprint(name)
        myprint(plist)
        conditional_pattern_base[item] = plist

    myprint(conditional_pattern_base)

    for item in sorted(conditional_pattern_base.keys(), key=lambda key: flist.index(key), reverse=True):
        temp_conditional_base = conditional_base + [item]
        PatternCombinations.append([header_table[item].frequency, temp_conditional_base])
        mine_fp_tree(conditional_pattern_base[item], temp_conditional_base, min_support)

def make_Frequent_Table(itemListCombinations, vocab_map):
    FrequentItems = {}
    for itemComb in itemListCombinations:
        item_freq = itemComb[0]
        if item_freq not in FrequentItems:
            FrequentItems[item_freq] = []
        itemset_words = []
        for item in itemComb[1]:
            itemset_words.append(vocab_map[item])
        FrequentItems[item_freq].append(itemset_words)
    return FrequentItems

def findSuperSet(pattern, patterns, supportCheck=False):
    for pat in patterns:
        match = 1
        for spat in pattern[1]:
            if not spat in pat[1]:
                match = 0
                break
        if match == 1:
            if supportCheck and pattern[0] == pat[0]:
                return True
            elif supportCheck and pattern[0] != pat[0]:
                pass
            else:
                return True
    return False

def findMaxClosedPattern(FrequentPatterns, supportCheck=False):
    Patterns = {}
    for i in range(1, max(FrequentPatterns.keys()) + 1):
        for pattern in FrequentPatterns[i]:
            if (i+1) in FrequentPatterns and findSuperSet(pattern, FrequentPatterns[i + 1], supportCheck):
                pass
            else:
                if pattern[0] not in Patterns: Patterns[pattern[0]] = []
                Patterns[pattern[0]].append(pattern[1])
    return Patterns

def make_level_wise_pattern(FrequentPatterns):
    Patterns = {}
    for itemset_support in FrequentPatterns.keys():
        for pattern in FrequentPatterns[itemset_support]:
            length_pattern = len(pattern)
            if length_pattern not in Patterns: Patterns[length_pattern] = []
            Patterns[length_pattern].append([itemset_support, pattern])
    return Patterns

# def findFrequencyInFile(pattern, topic_pattern, transaction):
#     frequency_count = 0
#     spattern = set(pattern[1])
#     this_topic_pattern = [ptrn[0] for ptrn in topic_pattern if set(ptrn[1]).issuperset(spattern)]
#
#     if len(this_topic_pattern) != 0: return this_topic_pattern[0]
#
#     for tline in transaction:
#         sline = set(tline[1])
#         if sline.issuperset(spattern):
#             frequency_count += tline[0] #Increment By the time how many transactions may be present in file
#     return frequency_count
#
# def findPureItems(topic_patterns, topics_transactions):
#     topics_purity_items = []
#     pattern_transaction_frequency = {}
#     for i in range(0, len(topic_patterns)):
#         for pattern in topic_patterns[i]:
#             fcounts = []
#             for j in range(0, len(topics_transactions)):
#                 frequency_count = findFrequencyInFile(pattern, topic_patterns[j], topics_transactions[j])
#                 fcounts.append(frequency_count)
#             pattern_transaction_frequency[pattern] = [fcounts, pattern



def writePatternInFile(file_name, FrequentItems):
    if not os.path.exists(file_name):
        os.makedirs(file_name)
    f = open(file_name + "/" + file_name + "-" + str(num_file) + ".txt", 'w')
    for FI in sorted(FrequentItems.keys(), key=lambda key: key, reverse=True):
        for patterns in FrequentItems[FI]:
            line = str(FI) + " ["
            for pattern in patterns:
                line = line + str(pattern) + ", "
            line = line[:-2]
            line += "]\n"
            f.write(line)
    f.close()

def mine_frequent_patterns(transactions, vocab_map, num_file, conditional_base = [], min_support = 1, make_graph=False):
    global PatternCombinations
    PatternCombinations = []
    mine_fp_tree(transactions, conditional_base, min_support, make_graph)
    print PatternCombinations
    print len(PatternCombinations)

    return PatternCombinations

def solveProblem(topics_transactions, num_files, vocab_map, min_support, make_graph=False):

    topics_patterns = []
    for num_file in range(0, num_files):
        pattern_combinations = mine_frequent_patterns(topics_transactions[num_file], vocab_map, num_file, [], min_support, make_graph)
        FrequentItems = make_Frequent_Table(pattern_combinations, vocab_map)
        Patterns = make_level_wise_pattern(FrequentItems)
        print FrequentItems
        print Patterns
        writePatternInFile("pattern", FrequentItems)
        maxFrequentPatterns = findMaxClosedPattern(Patterns)
        writePatternInFile("max", maxFrequentPatterns)
        closedFrequentPatterns = findMaxClosedPattern(Patterns, True)
        writePatternInFile("closed", closedFrequentPatterns)
        topics_patterns.append(pattern_combinations)

    # topics_purity_items = findPureItems(topics_patterns, topics_transactions, num_file)
    # for i in range(0, len(topics_purity_items)):
    #     pass
        #writePatternInFile("purity", topics_purity_items[i])

def readTransactionsFromFile(path, num_file):
    transactions = []
    file_name = path + "/" + "topic-" + str(num_file) + ".txt"
    split_point = re.compile("\s+")
    f = open(file_name, 'r')
    for line in f:
        line = line.strip()
        words = split_point.split(line)
        num_list = []
        for word in words:
            num_list.append(int(word))
        transactions.append([1, num_list])
    f.close()
    return transactions

def makeVocab(path):
    vfile_name = path + "/" + "vocab.txt"
    split_point = re.compile("\s+")
    vocab_map = {}
    f = open(vfile_name, 'r')
    for line in f:
        line = line.strip()
        words = split_point.split(line)
        vocab_map[int(words[0])] = words[1]
    f.close()
    return vocab_map


if __name__ == '__main__':
    min_support = 20
    path = "/home/manshu/UIUC/CS 412 - Data Mining/data-assign3/data-assign3"
    if len(sys.argv) >= 2:
        path = sys.argv[1]
    transactions = [
        [1, ['f', 'a', 'c', 'd', 'g', 'i', 'm', 'p']],
        [1, ['a', 'b', 'c', 'f', 'l', 'm', 'o']],
        [1, ['b', 'f', 'h', 'j', 'o', 'w']],
        [1, ['b', 'c', 'k', 's', 'p']],
        [1, ['a', 'f', 'c', 'e', 'l', 'p', 'm', 'n']]
    ]
    make_graph = False
    if make_graph:
        import ubigraph
        U = ubigraph.Ubigraph()
        U.clear()
        #rrgraph_style = U.newVertexStyle(shape="sphere", color="#ffff00", size="2.0")

    FpDt = [
            [10047, 17326, 17988, 17999, 17820],
            [17326, 9674, 17446, 17902, 17486],
            [17988, 17446, 9959, 18077, 17492],
            [17999, 17902, 18077, 10161, 17912],
            [17820, 17486, 17492, 17912, 9845]
        ]

    vocab_map = makeVocab(path)
    ts1 = time.time()
    num_files = 5
    topics_transactions = []

    for num_file in range(0, num_files):
        ptransactions = readTransactionsFromFile(path, num_file)
        topics_transactions.append(ptransactions)

    solveProblem(topics_transactions, num_files, vocab_map, min_support, make_graph)

    #mine_frequent_patterns(transactions, vocab_map, 0, [], min_support, make_graph)
    ts2 = time.time()
    print ts2 - ts1