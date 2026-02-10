def read_input():
    grid = []
    print("Enter the Sudoku puzzle row by row (use 0 for empty cells):")
    for _ in range(9):
        row_str = input().strip()
        if len(row_str) != 9:
            raise ValueError("Each row must have exactly 9 digits")
        row = [int(ch) for ch in row_str]  
        grid.append(row)
    return grid


def print_sudoku(grid):
    for row in range(9):
        if row % 3 == 0 and row != 0:
            print("-" * 21)  # horizontal separator

        for col in range(9):
            if col % 3 == 0 and col != 0:
                print("|", end=" ")

            if grid[row][col] == 0:
                print(".", end=" ")  # show empty cells as dots
            else:
                print(grid[row][col], end=" ")
        print()  # new line after each row


def find_empty(grid):
    for row in range(9):
        for col in range(9):
            if grid[row][col] == 0:
                return row, col
    return None


def is_valid_move(grid, num, pos):
    # Check row
    for col in range(9):
        if grid[pos[0]][col] == num and col != pos[1]:
            return False

    # Check column
    for row in range(9):
        if grid[row][pos[1]] == num and row != pos[0]:
            return False

    # Check 3x3 box
    box_x = pos[1] // 3
    box_y = pos[0] // 3
    for i in range(box_y * 3, box_y * 3 + 3):
        for j in range(box_x * 3, box_x * 3 + 3):
            if grid[i][j] == num and (i, j) != pos:
                return False

    return True

#recursive backtracking solver
def solve_sudoku(grid):
    empty = find_empty(grid)
    if not empty:
        return True  
    row, col = empty

    for num in range(1, 10):
        if is_valid_move(grid, num, (row, col)):
            grid[row][col] = num

            if solve_sudoku(grid):
                return True

            grid[row][col] = 0

    return False


# main program
sudoku_grid = read_input()
print("\nInitial Sudoku:")
print_sudoku(sudoku_grid)

if solve_sudoku(sudoku_grid):
    print("\nSolved Sudoku:")
    print_sudoku(sudoku_grid)
else:
    print("No solution exists for the given puzzle.")