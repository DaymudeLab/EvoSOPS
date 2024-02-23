import sys

def print_colored_number(number):
    color_mapping = {
        0: '\033[30m',  # Black
        1: '\033[33m',  # Yellow
        2: '\033[34m',  # Blue
        3: '\033[31m',  # Red
        4: '\033[35m',  # Purple
        5: '\033[30m'   # Black
    }

    # Reset color after printing
    reset_color = '\033[0m'

    if number in color_mapping:
        color_code = color_mapping[number]
        print(f"{color_code}{number}{reset_color}", end="")
    else:
        print(number)

def main():
    args = sys.argv

    if len(args) != 2:
        print("Invalid amount of arguments.")
        exit()

    filepath = args[1]

    with open(filepath, "r") as file: 
        for line in file:
            for character in line.strip().split("  "):
                print_colored_number(int(character))
                print("  ", end="")
            print()

if __name__=="__main__":
    main()