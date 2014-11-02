__author__ = 'sharma55'

import Queue
import itertools
import re
import os
import time
import math
import sys
import shutil
import errno

## FP Tree node containing its childs, parent, its name and prefix value
class TreeNode:
    def __init__(self, name, fq_value):
        self.childs = {}
        self.parent = None
        self.name = name
        self.fq_value = fq_value
        self.graph_node = None## This is used for graph visualization

# Header table class having item, frequency and head to their linked list pointer to an occurence of item in fp tree
class HeaderTable:
    def __init__(self, item, frequency):
        self.item = item
        self.frequency = frequency
        self.head = None

# General Linked List class having a value and next pointer
class LinkedListNode:
    def __init__(self, Node, value):
        self.value = value
        self.node = Node
        self.next = None

## Make fp tree from the ordered transactions and also adjust pointers to the head in header table
def make_fp_tree(transactions, header_table):
    root = TreeNode("{null}", 0) # create null root at the top
    if make_graph: ## for visualization purpose
        Groot = U.newVertex(shape="sphere", color="#ffff00", size="1.5", label=str(root.name), fontsize="32")
        root.graph_node = Groot
    ## insert nodes in the tree according to the prefix shared
    for transaction in transactions:
        temp_root = root
        if make_graph: Gtemp_root = Groot
        trans_freq = transaction[0]
        for i in range(0, len(transaction[-1])):
            item = transaction[-1][i]
            if (temp_root.childs.has_key(item)):
                temp_root.childs[item].fq_value += trans_freq  ### if item present in same location in the tree increase its frequency
                if make_graph: temp_root.childs[item].graph_node.set(label=str(item) + " " + str(temp_root.childs[item].fq_value))
            else:
                temp_root.childs[item] = TreeNode(item, trans_freq)  ### if item not present in this location insert a new node
                temp_root.childs[item].parent = temp_root  ## adjust parent pointers to find transaction path later when required
                ### adjust linked list pointer of the item in Header table
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
            ## change root to the current child to create nodes further in order
            temp_root = temp_root.childs[item]
    return root

# print fp tree for debugging
def myprint_fp_tree(tree_root):
    queue = Queue.Queue()
    queue.put(tree_root)
    while not queue.empty():
        node = queue.get()
        myprint(str(node.name) + " " + str(node.fq_value))
        for child in node.childs.keys():
            queue.put(node.childs[child])

## Order the transactions in descending manner according to their order in flist
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

## Calculate frequency of each item in the transactions and keep only those whose frequency of occurence > min_support
def make_flist(transactions, support):
    header_table = {}
    for transaction in transactions:
        trans_freq = transaction[0]
        for item in transaction[-1]:
            if (item in header_table):
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

## check if tree has no branching and each node only has one child
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

# For the single branch tree return all possible 1, 2, 3, .... n(height of single branch tree) combinations of patterns in sorted fashion
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
            itemListCombinations.append([min_support, tuple(sorted(itemcom))])
    return itemListCombinations

def myprint(string):
    pass#print string #its a debugging print function

# mine fp tree for frequent patterns
def mine_fp_tree(transactions, conditional_base = (), min_support = 1, make_graph=False):
    header_table = make_flist(transactions, min_support) #create base header table by using flist
    flist = sorted(header_table.keys(), key=lambda key: header_table[key].frequency, reverse=True) #make flist from header table keys

    for key in flist:
        myprint(str(key) + " " + str(header_table[key].frequency))
    transactions = make_ordered_transaction(transactions, flist) #order each transaction according to flist
    # for transaction in transactions:
    #     myprint(transaction)
    tree = make_fp_tree(transactions, header_table) # make fp tree using ordered transactions and adjust header table
    myprint_fp_tree(tree)
    myprint("=========================================================")
    myprint("=========================================================")

    # check if the tree has only branch and mining for this chain of conditional based can be stopped further
    # get patterns from all 1,2...n(height of single branch tree) combinations of items in this tree in sorted manner
    # combine all those patterns with conditional_base.
    # Put the combined items in global PatternCombinations and return
    if (checkPatternEnding(tree)):
        item_combinations = giveCombinations(tree)
        if item_combinations:
            #print item_combinations
            for item_list in item_combinations:
                temp_conditional_base = tuple(sorted(item_list[1] + conditional_base))
                PatternCombinations.append([item_list[0], temp_conditional_base])
        return

    if (len(tree.childs) == 0): return
    #
    # for item in header_table:
    #     string = str(header_table[item].item) + " " + str(header_table[item].frequency)
    #     temp_node = header_table[item].head
    #     while (temp_node):
    #         string += " " + str(temp_node.value) + " " + str(len(temp_node.node.childs))
    #         temp_node = temp_node.next
    #     myprint(string)

    # To create conditional pattern base table for each item in the header table create its further transaction table to mine it
    # Use linked list pointer in header table to point to a first occurence of item in fp tree, find its transaction path using parent pointers and
    # then move the linked list pointer to next occurence of item, find its transaction path and do the same till the linked list ends in null value.
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

    # for each conditional base add the item to the current conditional base and add to global Patterns
    # Now mine on this conditional fp tree recursively
    for item in sorted(conditional_pattern_base.keys(), key=lambda key: flist.index(key), reverse=True):
        temp_conditional_base = conditional_base + (item, )
        temp_conditional_base = tuple(sorted(temp_conditional_base))
        PatternCombinations.append([header_table[item].frequency, temp_conditional_base])
        mine_fp_tree(conditional_pattern_base[item], temp_conditional_base, min_support)

# make frequent pattern table in the dictionary format like this {support1 : [pattern1, pattern2], support 2 : [pattern3, pattern4, pattern5]} for easy access.
# Also, pattern is converted to word from numbers using vocab map
def make_Frequent_Table(itemListCombinations, vocab_map):
    FrequentItems = {}
    for itemComb in itemListCombinations:
        item_freq = itemComb[0]
        if item_freq not in FrequentItems:
            FrequentItems[item_freq] = []
        itemset_words = ()
        for item in itemComb[1]:
            itemset_words += (vocab_map[item], )
        FrequentItems[item_freq].append(itemset_words)
    return FrequentItems

# make patterns in dictionary format from the list format for easy access like {pattern1 : support1, pattern2 : support1, pattern3 : support2}
def make_Pattern_Dictionary(itemListCombinations, vocab_map):
    PatternDict = {}
    for itemComb in itemListCombinations:
        item_freq = itemComb[0]
        itemset_words = ()
        for item in itemComb[1]:
            itemset_words += (vocab_map[item], )
        if itemset_words not in PatternDict:
            PatternDict[itemset_words] = []
        PatternDict[itemset_words] = item_freq
    return PatternDict


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

## Here we create a level wise patterns for easy access in max and closed patterns
## For example { 1 : [[sup1, freq-itemset-1], [sup2, freq-itemset-1] ..... ],
##               2 : [[sup3, freq-itemset-2], [sup4, freq-itemset-2] ..... ],
##               ..................................... }
def make_level_wise_pattern(FrequentPatterns):
    Patterns = {}
    for itemset_support in FrequentPatterns.keys():
        for pattern in FrequentPatterns[itemset_support]:
            length_pattern = len(pattern)
            if length_pattern not in Patterns: Patterns[length_pattern] = []
            Patterns[length_pattern].append([itemset_support, pattern])
    return Patterns

## find frequency in other topics, if present in the global topic_patterns containing frequent patterns of all files then return it, otherwise 0
def findFrequencyInFile(pattern, topic_pattern, transaction):
    frequency_count = 0

    if pattern in topic_pattern: return topic_pattern[pattern]

    # for tline in transaction:
    #     match = 1
    #     for word in pattern:
    #         if word not in tline:
    #             match = -1
    #             break
    #     if match == 1:
    #         frequency_count += 1

    return frequency_count

## make ftp table for purity calculation
def makeFtpTable(topic_patterns, topics_transactions):
    pattern_transaction_frequency = {}
    for i in range(0, len(topic_patterns)):
        for pattern in topic_patterns[i].keys():
            if pattern in pattern_transaction_frequency:
                continue
            fcounts = []
            for j in range(0, len(topics_transactions)):
                frequency_count = findFrequencyInFile(pattern, topic_patterns[j], topics_transactions[j])
                fcounts.append(frequency_count)
            pattern_transaction_frequency[pattern] = [topic_patterns[i][pattern], fcounts]
    return pattern_transaction_frequency

## write Purity values in file
def writePurityInFile(file_name, num_file, PurityItems):
    if not os.path.exists(file_name):
        os.makedirs(file_name)
    f = open(file_name + "/" + file_name + "-" + str(num_file) + ".txt", 'w+')
    for FI in sorted(PurityItems.keys(), key=lambda key: key, reverse=True):
        #PurityItems[FI].sort(reverse=True)
        for patterns in PurityItems[FI]:#sorted(PurityItems[FI], key=lambda key: PurityItems[FI][0], reverse=True):
            #line = str(FI) + " ["
            line = str(FI) + " " #str(patterns[0]) + " "
            for pattern in patterns[1]:
                line = line + str(pattern) + " "
            #line = line[:-2]
            line += "\n"
            f.write(line)
    f.close()

## write double sorted patterns in the file.
def writeDoubleSortedPatternInFile(file_name, num_file, PatternItems):
    if not os.path.exists(file_name):
        os.makedirs(file_name)
    f = open(file_name + "/" + file_name + "-" + str(num_file) + ".txt", 'w+')
    for FI in sorted(PatternItems.keys(), key=lambda key: key, reverse=True):
        PatternItems[FI].sort(reverse=True)
        for patterns in PatternItems[FI]:
            line = str(FI) + " ["
            for pattern in patterns[1]:
                line = line + str(pattern) + ", "
            line = line[:-2]
            line += "]\n"
            f.write(line)
    f.close()

## find pure items in all the topics
def findPureItems(topic_patterns, topics_transactions, Dt):
    ftp_table = makeFtpTable(topic_patterns, topics_transactions)
    #print ftp_table
    num_topics = len(topics_transactions)
    for num_file in range(0, num_topics):
        PurityItems = {}
        num_items_in_transactions = len(topics_transactions[num_file])
        for pattern in topic_patterns[num_file].keys():
            ftp = ftp_table[pattern][1]
            max_val = max([((ftp[num_file] + ftp[t]) / (1.0 * Dt[num_file][t])) for t in range(0, num_topics) if t != num_file])
            purity_value = math.log(ftp[num_file] / (1.0 * Dt[num_file][num_file])) - math.log(max_val)
            purity_value = round(purity_value, 4)
            #if purity_value not in PurityItems:
            #    PurityItems[purity_value] = []
            #PurityItems[purity_value].append([topic_patterns[num_file][pattern], pattern])

            ranking_measure = purity_value * (topic_patterns[num_file][pattern] / (1.0 * num_items_in_transactions))
            ranking_measure = round(ranking_measure, 6)
            if ranking_measure not in PurityItems:
                PurityItems[ranking_measure] = []
            #PurityItems[ranking_measure].append([topic_patterns[num_file][pattern], pattern])
            PurityItems[ranking_measure].append([purity_value, pattern])
            #print "Purity = ", PurityItems

        writePurityInFile("purity", num_file, PurityItems)

## write patterns, max patterns and closed patterns into file by descending supported according to their name
def writePatternInFile(file_name, num_file, FrequentItems):
    if not os.path.exists(file_name):
        os.makedirs(file_name)
    f = open(file_name + "/" + file_name + "-" + str(num_file) + ".txt", 'w+')
    for FI in sorted(FrequentItems.keys(), key=lambda key: key, reverse=True):
        for patterns in FrequentItems[FI]:
            line = str(FI) + " "
            for pattern in patterns:
                line = line + str(pattern) + " "
            #line = line[:-2]
            line += "\n"
            f.write(line)
    f.close()

## call mine_fp_tree with null conditional base
def mine_frequent_patterns(transactions, conditional_base = (), min_support = 1, make_graph=False):
    global PatternCombinations
    PatternCombinations = [] #all patterns from each recursive tree will be accumulated in this global variable
    mine_fp_tree(transactions, conditional_base, min_support, make_graph) #mine fp tree with empty conditional base with passed min_support

    return PatternCombinations

## find coverage of patterns
def findCoveragePattern(FrequentPatterns, num_transactions):
    Patterns = {}
    for pattern in FrequentPatterns:
        coverage_value = FrequentPatterns[pattern] / (1.0 * num_transactions)
        coverage_value = round(coverage_value, 4)
        if coverage_value not in Patterns:
            Patterns[coverage_value] = []
        Patterns[coverage_value].append(pattern)
    return Patterns

## find phraseness of patterns
def findPhrasenessPattern(FrequentPatterns, num_transactions):
    Patterns = {}
    for pattern in FrequentPatterns:
        coverage_value = FrequentPatterns[pattern] / (1.0 * num_transactions)
        phraseness_value = math.log(coverage_value)
        if len(pattern) > 1:
            sum_word_prob = 0
            for word in pattern:
                sum_word_prob += math.log(FrequentPatterns[(word, )] / (1.0 * num_transactions))
            phraseness_value -= sum_word_prob

        phraseness_value = round(phraseness_value, 4)

        if phraseness_value not in Patterns:
            Patterns[phraseness_value] = []
        Patterns[phraseness_value].append([FrequentPatterns[pattern], pattern])
    return Patterns

## find completeness of patterns
def findCompletePattern(FrequentPatterns, Single_Words_Patterns, vocab_map):
    Patterns = {}
    for pattern in FrequentPatterns:
        max_val = 0
        for word in Single_Words_Patterns:
            if word not in pattern:
                another_pattern = tuple(sorted(pattern + (word, )))
                if another_pattern not in FrequentPatterns:
                   continue
                else:
                    if max_val < FrequentPatterns[another_pattern]:
                        max_val = FrequentPatterns[another_pattern]
            else:
                continue
        completeness_value = 1.0 - (max_val / (1.0 * FrequentPatterns[pattern]))
        completeness_value = round(completeness_value, 4)
        if completeness_value not in Patterns:
            Patterns[completeness_value] = []
        word_pattern = tuple()
        for num in pattern:
            word_pattern += (vocab_map[num], )
        Patterns[completeness_value].append([FrequentPatterns[pattern], word_pattern])

    return Patterns

## solve assignment problem
def solveProblem(topics_transactions, word_transactions, num_files, vocab_map, FpDt, rel_min_support, make_graph=False):
    topics_patterns = []
    rev_vocab_map = {vocab_map[num_word] : num_word for num_word in vocab_map}

    for num_file in range(0, num_files):
        print "\n=========================================================\n"
        min_support = rel_min_support * len(topics_transactions[num_file]) #convert relative support to absolute support
        ##########################################################
        print "Mining fp tree for topic-" + str(num_file)
        pattern_combinations = mine_frequent_patterns(topics_transactions[num_file], (), min_support, make_graph) ## mine_frequent_patterns using FP-Tree
        ##########################################################
        FrequentItems = make_Frequent_Table(pattern_combinations, vocab_map)
        PatternDict = make_Pattern_Dictionary(pattern_combinations, vocab_map)
        Patterns = make_level_wise_pattern(FrequentItems)
        if 3 in Patterns:
            print "Number of 3-word-itemsets = ", len(Patterns[3])
        else:
            print "Number of 3-word-itemsets = 0"
        ## Append dictionary of frequent patterns of each file in one variable for easy calculation of purity
        topics_patterns.append(PatternDict)

        print "Length of frequent items in topic-" + str(num_file) + " = ", len(PatternCombinations)
        #print "Pattern Combinations = ", pattern_combinations
        #print "Frequent Items = ", FrequentItems
        #print "Pattern Dictionary of frequent items in topic-" + str(num_file) + " = ", PatternDict
        #print "Patterns = ", Patterns

        writePatternInFile("pattern", num_file, FrequentItems) # write patterns into file

        print "Finding max patterns for topic-" + str(num_file)
        maxFrequentPatterns = findMaxClosedPattern(Patterns) # find max patterns
        writePatternInFile("max", num_file, maxFrequentPatterns) # write max patterns

        print "Finding closed patterns for topic-" + str(num_file)
        closedFrequentPatterns = findMaxClosedPattern(Patterns, True) # find closed patterns
        writePatternInFile("closed", num_file, closedFrequentPatterns) # write closed patterns

        print "Finding coverage of patterns for topic-" + str(num_file)
        coverageOfFrequentItems = findCoveragePattern(PatternDict, len(topics_transactions[num_file])) # find coverage of patterns
        writePatternInFile("coverage", num_file, coverageOfFrequentItems) # write coverage of patterns

        print "Finding phraseness of patterns for topic-" + str(num_file)
        phrasenessOfFrequentItems = findPhrasenessPattern(PatternDict, len(topics_transactions[num_file])) # find phraseness of patterns
        writeDoubleSortedPatternInFile("phraseness", num_file, phrasenessOfFrequentItems) # write phraseness of patterns

        print "Finding completeness of patterns for topic-" + str(num_file)
        # find completeness using single
        Single_Num_Word_Patterns = [rev_vocab_map[word_comb[1][0]] for word_comb in Patterns[1]]
        Num_Patterns = {word_comb[1]: word_comb[0] for word_comb in pattern_combinations}
        completenessOfFrequentItems = findCompletePattern(Num_Patterns, Single_Num_Word_Patterns, vocab_map)
        writeDoubleSortedPatternInFile("completeness", num_file, completenessOfFrequentItems)

    if num_files > 1:
        print "Finding Purity of patterns for all patterns for all topics"
        ## if files > 1, then find purity values for each pattern
        findPureItems(topics_patterns, word_transactions, FpDt) #write will be called in this function only
    else:
        print "Cannot call purity for just a single topic."

# read transactions from file in numbers and word format
def readTransactionsFromFile(path, num_file, vocab_map):
    print "Reading transactions from topic-" + str(num_file) + ".txt"
    transactions = []
    word_transactions = []
    file_name = path + "/" + "topic-" + str(num_file) + ".txt"
    split_point = re.compile("\s+")
    f = open(file_name, 'r')
    for line in f:
        line = line.strip()
        words = split_point.split(line)
        num_list = []
        word_list = []
        for word in words:
            topic_name = int(word)
            num_list.append(topic_name)
            word_list.append(vocab_map[topic_name])
        transactions.append([1, num_list])
        word_transactions.append(word_list)
    f.close()
    return transactions, word_transactions

# make dictionary to map number to words
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
    min_support = 0.0025 #Relative min support
    path = "." #"/home/manshu/UIUC/CS 412 - Data Mining/data-assign3/data-assign3"
    if len(sys.argv) >= 2:
        path = sys.argv[1]
    if len(sys.argv) >= 3:
        min_support = float(sys.argv[2])
    # take path from arguments
    # Dummy transactions for easy visualization
    # transactions = [
    #     [1, ['f', 'a', 'c', 'd', 'g', 'i', 'm', 'p']],
    #     [1, ['a', 'b', 'c', 'f', 'l', 'm', 'o']],
    #     [1, ['b', 'f', 'h', 'j', 'o', 'w']],
    #     [1, ['b', 'c', 'k', 's', 'p']],
    #     [1, ['a', 'f', 'c', 'e', 'l', 'p', 'm', 'n']]
    # ]
    ### Make graph varibale for visualization. Requires graph packages to be installed
    make_graph = False

    if make_graph:
        import ubigraph
        U = ubigraph.Ubigraph()
        U.clear()
        #graph_style = U.newVertexStyle(shape="sphere", color="#ffff00", size="2.0")

    # D(t) given in the assignment itself
    FpDt = [
            [10047, 17326, 17988, 17999, 17820],
            [17326, 9674, 17446, 17902, 17486],
            [17988, 17446, 9959, 18077, 17492],
            [17999, 17902, 18077, 10161, 17912],
            [17820, 17486, 17492, 17912, 9845]
        ]

    # make a dictionary to map numbers to their corresponding topics
    vocab_map = makeVocab(path)
    num_files = 5 #num of topic files
    topics_transactions = []
    topics_words_transactions = []

    ts1 = time.time()

    for num_file in range(0, num_files):
        ptransactions, word_transactions = readTransactionsFromFile(path, num_file, vocab_map) # read each topic_file one by one and return give transactions in word or integer format
        topics_transactions.append(ptransactions) #keep a track of all transactions
        topics_words_transactions.append(word_transactions)

    #solve the assignment problems
    solveProblem(topics_transactions, topics_words_transactions, num_files, vocab_map, FpDt, min_support, make_graph)

    #mine_frequent_patterns(transactions, (), min_support, make_graph)

    #shutil.copytree("pattern", "patterns") # copy contents of pattern into patterns, since not clear in assignment that folder should be pattern or patterns
    src, dst = ("pattern", "patterns")
    try:
        shutil.rmtree(dst)
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else: raise
    ts2 = time.time()
    print "\n\nFull time taken", ts2 - ts1
