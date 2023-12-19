# Reverse Map: index in genome's dimension -> various intra group(F/M/B) configurations 
# intra group(F/M/B) configurations ie. all_cnt, same_clr_cnt, all_possible_cnt(static position cnt in F/M/B — (3/2/3))
# reverse_idx_config_map = [(0,0,3),(1,1,3),(1,0,3),(2,2,3),(2,1,3),(2,0,3),(3,3,3),(3,2,3),(3,1,3),(3,0,3)]
prob_cache = {}
prob_set = []

## Theory defined
# LAMBDA = 4
LAMBDA = 6
# GAMMA = 6

# Genome[back_cnt][mid_cnt][front_cnt]]
idx = 1
arr = [[[0 for k in range(4)] for j in range(3)] for i in range(4)]
for n in range(4):
    for j in range(3):
        for i in range(4):
            # ignoring j (for mid group) since it doesn't change the gain/loss in neighbors
            # (e, es, _) = reverse_idx_config_map[n]
            # (e_, es_, __) = reverse_idx_config_map[i]
            
            # prob = round(min(LAMBDA**(i - n),1),4)
            prob = round(min(LAMBDA**(-1 * (n + j)),1),4)
            if prob_cache.get(str(prob)):
                pass
            else:
                prob_cache[str(prob)] = idx
                idx += 1
                prob_set.append(prob)
            arr[n][j][i] = (prob_cache[str(prob)] - 1)

print(arr)
# print(prob_cache)
print(len(prob_cache.keys()))
print([int(x*10000) for x in prob_set])