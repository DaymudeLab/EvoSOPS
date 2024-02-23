import sys
import re, math

def main():
    args = sys.argv

    if len(args) != 2:
        print("Invalid amount of arguments.")
        exit()

    filepath = args[1]

    fitness = {}

    for i in range(10):
        fitness[i] = 0

    with open(filepath, "r") as file:
        for line in file:
            s = re.search("Avg\. Fitness -> ([0-9.]*)", line)
            if s:
                num = float(s.group(1))
                fitness[math.floor((num * 10))] += 1

    for key in fitness.keys():
        print(f".{key}:{fitness[key]}")

if __name__=="__main__":
    main()