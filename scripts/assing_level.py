import numpy as np
def read_inter_matrix(datafile="levels.dat"):
    with open(datafile, 'r') as _file:
        lines = _file.readlines()
        level_dict = {}
        col_labels = lines[0].strip().split() 
        bead_to_index = dict(zip(col_labels, np.arange(0, len(col_labels))))
        for line in lines[1:]:
            tokens = line.strip().split()
            row_type = tokens[-1]
            for idx, token in enumerate(tokens[:-1]):
                level_dict[frozenset([row_type, col_labels[idx]])] = int(token)
    return level_dict, bead_to_index

def modify_level(level, ba, bb, alA, alB, dlA, dlB, rlA, rlB, hlA, hlB):
    if ba == 'W' and (alB or dlB):
        level += 1
    if bb == 'W' and (alA or dlA):
        level += 1
    if alA and alB:
        level += 1
    if dlA and dlB:
        level += 1
    if (alA and dlB) or (dlA and alB):
        level -= 1
    if rlA and rlB:
        level += 1
    if hlA and hlB:
        level -= 1
    return level

def assign_levels(bead_types_a, bead_types_b):
    """
    Assign level between beadtypes.
    """
    bead_matrix, bead_to_index = read_inter_matrix('levels.dat')
    size_to_plane = {frozenset((0, 0)): 0,
                     frozenset((1, 1)): 1,
                     frozenset((2, 2)): 2,
                     frozenset((0, 1)): 3,
                     frozenset((0, 2)): 4,
                     frozenset((1, 2)): 5,}

    for bead_a, bead_b in zip(bead_types_a, bead_types_b):
        sizes = []
        types_clean = []
        labels = {}
        o_bead = [bead_a, bead_b]
        for tag, bead in zip(['A','B'], [bead_a, bead_b]):
            if bead[0] == "T":
                size = 0
                bead = bead[1:]
            elif bead[0] == "S":
                size = 1
                bead = bead[1:]
            else:
                size = 2
            sizes.append(size)
            al = False
            dl = False
            hl = False
            rl = False
            el = False
            vl = False
            if bead[-1].isupper() and bead[-1] != 'W':
                bead = bead[:-1]
            if bead[-1] == "r":
                rl = True
                bead = bead[:-1]
            if bead[-1] == "h":
                hl = True
                bead = bead[:-1]
            if bead[-1] == 'e':
                el = True
                bead = bead[:-1]
            if bead[-1] == 'v':
                vl = True
                bead = bead[:-1]
            if bead[-1] == "a":
                al = True
                bead = bead[:-1]
            if bead[-1] == "d":
                dl = True
                bead = bead[:-1]
            # we drop e/v labels here because they don't affect this paticular system
            labels.update({f"rl{tag}":rl, f"hl{tag}": hl, f"al{tag}": al, f"dl{tag}": dl})
            types_clean.append(bead)

        level = bead_matrix[frozenset(types_clean)]
        level = modify_level(level=level,
                             ba=types_clean[0],
                             bb=types_clean[1],
                             **labels)
        third_dimension = size_to_plane[frozenset(sizes)]
        yield o_bead[0], o_bead[1], level, third_dimension

# for bead_a, bead_b, level, third_dimension in assign_levels(["SP4r", "SP2d", "SP2", "TP3A"], ["SP1d", "TP1a", "N6dA"]):
    # print(bead_a, bead_b, level, third_dimension)
