g_ga = [[[4, 0, 0, 1], [6, 1, 0, 0], [1, 6, 10, 0]], [[7, 1, 2, 2], [3, 1, 0, 1], [4, 9, 5, 0]], [[6, 1, 2, 0],[10, 10, 0, 1], [10, 2, 9, 8]], [[8, 7, 3, 0], [9, 9, 7, 8], [7, 4, 7, 2]]]
g_theory = [[[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]], [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]], [[2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]], [[3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5]]]


prob_ga = [1000, 500, 250, 125, 63, 31, 16, 8, 4, 2, 1]
prob_theory = [1000, 167, 28, 5, 1, 1]
# reverse_idx_fb_map = [(0,0,3),(1,1,3),(1,0,3),(2,2,3),(2,1,3),(2,0,3),(3,3,3),(3,2,3),(3,1,3),(3,0,3)]
# reverse_idx_m_map = [(0,0,2),(1,1,2),(1,0,2),(2,2,2),(2,1,2),(2,0,2)]

# Genome[back_cnt][mid_cnt][front_cnt]]
gene = 1
sig_diff = 0
diff_buck = [0]*10
ncnt_ga_buck = {}
ncnt_th_buck = {}

for n in range(4):
    for j in range(3):
        for i in range(4):
            # ignoring j (for mid group) since it doesn't change the gain/loss in neighbors
            # (b, bs, _) = reverse_idx_fb_map[n]
            # (f, fs, __) = reverse_idx_fb_map[i]
            # (m, ms, ___) = reverse_idx_m_map[j]
            pth = prob_theory[g_theory[n][j][i]]
            pga = prob_ga[g_ga[n][j][i]]
            diff = abs(pth-pga)
            ndiff = (i - n)
            nsum = (n + j) # e

            ncnt_ga_buck[str(nsum)] = ncnt_ga_buck.get(str(nsum), []) + [pga]
            ncnt_th_buck[str(nsum)] = ncnt_th_buck.get(str(nsum), []) + [pth]
            # ncnt_th_buck.get(str(ndiff), []).append(pth)

            print("Gene:",gene, "\tChange: ",ndiff, "\e: ",nsum)
            gene += 1
            if diff > 700:
                sig_diff += 1
                diff_buck[diff//100] += 1
                print("Back:{0:1d} Mid:{1:1d} Front:{2:1d}\nTheory:{3:d}\tGA:{4:d}\tDiff:{5:d}".format(n, j, i,pth, pga, diff))                
            print()

# print("\nSignificanlty diff genes:",sig_diff)
# print("\nSignificanlty diff range:",diff_buck)
print("\nNeigbor Count -> Probs GA:", ncnt_ga_buck)
print("\nNeigbor Count -> Probs Theory:", ncnt_th_buck)
