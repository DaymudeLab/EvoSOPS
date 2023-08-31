import itertools

groups = [
   [x for x in range(4)],
   [x for x in range(3)],
   [x for x in range(4)]
]
## state info based combination expansion
state_exp = [1, 2, 3, 4]

combs = list(itertools.product(*groups))
print("Total Configs:", len(combs))

state_config_cnt = 0
for comb in combs:
    state_info_combs_cnt = state_exp[comb[0]] * state_exp[comb[1]] * state_exp[comb[2]]
    state_config_cnt += state_info_combs_cnt
    print(comb, " requires -> ", state_info_combs_cnt, " state-combinations")

print("Total State Configurations:", state_config_cnt)
