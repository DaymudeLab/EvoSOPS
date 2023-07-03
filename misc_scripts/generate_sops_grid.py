arena_layers = 7
grid_size = (arena_layers*2 + 1)
grid = [[0]*grid_size for i in range(grid_size)]
for i in range(arena_layers):
    j = 1
    while i+arena_layers+j < grid_size:
        grid[i][(i+arena_layers+j)] = 2
        grid[(i+arena_layers+j)][i] = 2
        j +=1

for i in range(grid_size):
    print(grid[i])
print("\n")