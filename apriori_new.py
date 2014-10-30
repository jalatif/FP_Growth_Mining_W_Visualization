
import os
import math

# find frequent patterns values by comparing the candidate keys frequencies with minsup
def find_frequent(candidate, minsup, no_of_lines):
    a_dict = {}
    for key in candidate:
        if candidate[key] >= minsup * no_of_lines:
            a_dict[key] = candidate[key]
    return a_dict


# generate k-itemset candidates by joining the k-1-itemset candidates
def candidate_gen(keys):
    a_dict = {}
    for i in keys:
        l1_sz = len(i)
        for j in keys:
            found_super = 1
            for f in range(0, l1_sz - 1):
                if j[f] != i[f]:
                    found_super = 0
                    break
            if found_super == 1 and i[l1_sz - 1] < j[l1_sz - 1]:
                new_pattern = i + (j[l1_sz - 1], )
                a_dict[tuple(sorted(new_pattern))] = 0
    return a_dict


# get the frequency of each candidate comparing it with the original database transactions
def add_frequency(candidate, transactions):
    for pattern in candidate:
        for t in transactions:
            found = 1
            for item in pattern:
                if item not in t:
                    found = 0
                    break
            if found == 1:
                candidate[pattern] += 1
    return candidate


# read line by line from a file
def read_topic_file(i):
    return open("/home/manshu/UIUC/CS 412 - Data Mining/data-assign3/data-assign3/topic-" + (str(i)) + ".txt").read().split('\n')


def rev_freq(fs):
    rev = {}
    for F in fs:
        for pattern in F.keys():
            if F[pattern] not in rev:
                rev[F[pattern]] = []
            rev[F[pattern]].append(pattern)
    return rev


def vocab_map():
    f = open("/home/manshu/UIUC/CS 412 - Data Mining/data-assign3/data-assign3/vocab.txt", "r")
    v_map = {}
    import re
    match_point = re.compile("\s+")
    for line in f:
        words = match_point.split(line)
        v_map[words[0]] = words[1]

    return v_map

def apriori_generation(i, minsup, terms):
    transactions = []
    no_of_lines = 0

    for line in terms:
        split_line = line.split()
        # store all the transactions of the original database
        transactions.append(split_line)
        no_of_lines += 1
    no_of_lines -= 1

    # to generate all 1-itemset candidates
    input_file = open("/home/manshu/UIUC/CS 412 - Data Mining/data-assign3/data-assign3/topic-" + (str(i)) + ".txt", "r")
    c1 = {}
    f1 = {}
    for word in input_file.read().split():
        if word not in c1:
            c1[word] = 1
        else:
            c1[word] += 1
    for k, v in c1.items():
        if v >= minsup:
            f1.update({(k, ): v})

    # First iteration
    # frequent 1-itemset
    f1 = find_frequent(f1, minsup, no_of_lines)
    final_frequent_set = []
    f = f1
    final_frequent_set.append(f1)
    while len(f.keys()) != 0:
        c = candidate_gen(f.keys())
        if len(c) == 0:
            break
        c = add_frequency(c, transactions)
        # print len(c.keys())
        f = find_frequent(c, minsup, no_of_lines)
        #print f
        if len(f.keys()) != 0:
            final_frequent_set.append(f)
    print final_frequent_set

    # closed_pattern = closed_max_patterns(final_frequent_set)
    # print closed_pattern
    vm = vocab_map()
    rs = rev_freq(final_frequent_set)
    #print rs



    filename = "pattern/pattern-" + (str(i)) + ".txt"
    if not os.path.exists("pattern"):
        os.mkdir("pattern")
    out_file = open(filename, "w+")

    for support in sorted(rs, reverse=True):
        for pattern in rs[support]:
            words = []
            for word in pattern:
                words.append(vm[word])
            #print float(support)/float(no_of_lines), words
            out_file.write(str(float(support)/float(no_of_lines)) + " " + str(words) + "\n")
    out_file.close()

    max_pattern = convertFS(final_frequent_set)
    closed_pattern = convertFS(final_frequent_set, True)
    print len(max_pattern.keys())
    print max_pattern
    writePattern(max_pattern, "max", file_num, vm)

    print len(closed_pattern.keys())
    print closed_pattern
    writePattern(closed_pattern, "closed", file_num, vm)

    fs = {}
    for pl in final_frequent_set:
        for pattern in pl.keys():
            fs[pattern] = pl[pattern]
    return fs

def findSuperPattern(pattern, topics_patterns, pattern_support, supportCheck=False):
    for pt in topics_patterns.keys():
        found = 1
        for item in pattern:
            if item not in pt:
                found = 0
                break
        if found == 1:
            if topics_patterns[pt] > min_sup:
                if supportCheck:
                    if topics_patterns[pt] == pattern_support:
                        return True
                else:
                    return True
    return False

def convertFS(topics_fs, supportCheck = False):
    new_patterns = {}
    for i in range(0, len(topics_fs)):
        for pattern in topics_fs[i].keys():
            if (i == len(topics_fs) - 1) or not findSuperPattern(pattern, topics_fs[i+1], topics_fs[i][pattern], supportCheck):
                new_patterns[pattern] = topics_fs[i][pattern]
    return new_patterns

def rev_freq_pattern(fs):
    rev = {}
    for pattern in fs.keys():
        if fs[pattern] not in rev:
            rev[fs[pattern]] = []
        rev[fs[pattern]].append(pattern)
    return rev

def writePattern(Patterns, fname, i, vm):
    filename = fname + "/" + fname + "-" + (str(i)) + ".txt"
    if not os.path.exists(fname):
        os.mkdir(fname)
    out_file = open(filename, "w+")
    rs = rev_freq_pattern(Patterns)
    for support in sorted(rs, reverse=True):
        for pattern in rs[support]:
            words = []
            for word in pattern:
                words.append(vm[word])
            #print float(support)/float(no_of_lines), words
            out_file.write(str(support) + " " + str(words) + "\n")
    out_file.close()

def pure_items(all_patterns, D, vm):
    ftp = {}
    ltrans = len(all_patterns)
    for i in range(0, ltrans):
        ptrns = all_patterns[i]
        for j in range(0, ltrans):
            for ptrn in ptrns.keys():
                if ptrn not in ftp:
                    ftp[ptrn] = [0, []]
                    ftp[ptrn][0] = all_patterns[i][ptrn]
                    ftp[ptrn][1] = [0 for k in range(0, ltrans)]
                if ptrn not in all_patterns[j]:
                    ftp[ptrn][1][j] = 0
                else:
                    ftp[ptrn][1][j] = all_patterns[j][ptrn]
    print ftp

    for nfile in range(0, ltrans):
        PureItems = {}
        for ptr in all_patterns[nfile].keys():
            ftpp = ftp[ptr][1]
            lvalue = [((ftpp[nfile] + ftpp[i]) / (1.0 * D[nfile][i])) for i in range(0, ltrans) if i != nfile]
            maxVal = max(lvalue)
            purevalue = math.log(ftpp[nfile] / (1.0 * D[nfile][nfile])) - math.log(maxVal)
            purevalue = round(purevalue, 4)
            if purevalue not in PureItems:
                PureItems[purevalue] = []
            PureItems[purevalue].append([all_patterns[nfile][ptr], ptr])
        fname = "purity"
        filename = fname + "/" + fname + "-" + (str(nfile)) + ".txt"
        if not os.path.exists(fname):
            os.mkdir(fname)
        out_file = open(filename, "w+")
        for pi in sorted(PureItems.keys(), key=lambda key: key, reverse=True):
            PureItems[pi].sort(reverse=True)
            for ptrns in PureItems[pi]:
                content = str(pi) + " ["
                for ptr in ptrns[1]:
                    content += str(vm[ptr]) + ", "
                content = content[:-2]
                content += "]\n"
                out_file.write(content)
        out_file.close()


if __name__ == "__main__":
    min_sup = 0.01
    topics_patterns = []
    for file_num in range(0, 5):
        terms_in_topic = read_topic_file(file_num)
        fs = apriori_generation(file_num, min_sup, terms_in_topic)
        topics_patterns.append(fs)
    print topics_patterns

    D = [[10047, 17326, 17988, 17999, 17820],
        [17326, 9674, 17446, 17902, 17486],
        [17988, 17446, 9959, 18077, 17492],
        [17999, 17902, 18077, 10161, 17912],
        [17820, 17486, 17492, 17912, 9845]]
    pure_items(topics_patterns, D, vocab_map())