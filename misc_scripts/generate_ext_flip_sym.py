import itertools

# Define a list of numbers
loc = [x for x in range(8)]

flip_translation_equivalence = [0, 2, 1, 4, 3, 6, 5, 7]

def flip(combination):
    flipped_comb = []
    for elem in combination:
        flipped_elem = flip_translation_equivalence[elem]
        if flipped_elem in combination:
            flipped_comb.append(elem)
        else:
            flipped_comb.append(flip_translation_equivalence[elem])
    return tuple(flipped_comb)

# reduce the no. of combinations based on flip symmetry
def condense(combinations):
    reduced_combs = []
    base = combinations
    while len(base) > 0:
        comb = base.pop(0)
        reduced_combs.append(comb)
        flip_comb = flip(comb)
        try:
            base.remove(flip_comb)
        except ValueError:
            print("Following flip combo not found: ", flip_comb, " for: ", comb)
            ## Ignore
            continue
    return reduced_combs

for i in range(4+1):
    print()
    combs = list(itertools.combinations(loc, i))
    print("Total Combinations: ", len(combs))
    print(combs)
    print()
    cond_combs = condense(combs)
    print("Reduced Combinations: ", len(cond_combs))
    print(cond_combs)
    print()
