import math

class TicTacToe:
    def __init__(self):
        self.board = [' ' for _ in range(9)]  # Represents the 3x3 board
        self.current_winner = None

    def print_board(self):
        for row in [self.board[i*3:(i+1)*3] for i in range(3)]:
            print('| ' + ' | '.join(row) + ' |')

    @staticmethod
    def print_board_nums():
        number_board = [[str(i) for i in range(j*3, (j+1)*3)] for j in range(3)]
        for row in number_board:
            print('| ' + ' | '.join(row) + ' |')

    def available_moves(self):
        return [i for i, spot in enumerate(self.board) if spot == ' ']

    def empty_squares(self):
        return ' ' in self.board

    def num_empty_squares(self):
        return self.board.count(' ')

    def make_move(self, square, letter):
        if self.board[square] == ' ':
            self.board[square] = letter
            if self.winner(square, letter):
                self.current_winner = letter
            return True
        return False

    def winner(self, square, letter):
        # Check row
        row_ind = square // 3
        row = self.board[row_ind*3:(row_ind+1)*3]
        if all([spot == letter for spot in row]):
            return True
        # Check column
        col_ind = square % 3
        column = [self.board[col_ind+i*3] for i in range(3)]
        if all([spot == letter for spot in column]):
            return True
        # Check diagonal
        if square % 2 == 0:
            diagonal1 = [self.board[i] for i in [0, 4, 8]]
            if all([spot == letter for spot in diagonal1]):
                return True
            diagonal2 = [self.board[i] for i in [2, 4, 6]]
            if all([spot == letter for spot in diagonal2]):
                return True
        return False

def minimax(position, depth, maximizing_player, game):
    if depth == 0 or game.current_winner is not None:
        return position, evaluate_position(position, game)
    
    if maximizing_player:
        max_eval = -math.inf
        best_move = None
        for move in game.available_moves():
            game.make_move(move, 'O')
            eval_position = minimax(move, depth-1, False, game)[1]
            game.board[move] = ' '  # undo move
            if eval_position > max_eval:
                max_eval = eval_position
                best_move = move
        return best_move, max_eval
    else:
        min_eval = math.inf
        best_move = None
        for move in game.available_moves():
            game.make_move(move, 'X')
            eval_position = minimax(move, depth-1, True, game)[1]
            game.board[move] = ' '  # undo move
            if eval_position < min_eval:
                min_eval = eval_position
                best_move = move
        return best_move, min_eval

def evaluate_position(position, game):
    if game.winner(position, 'O'):
        return 1
    elif game.winner(position, 'X'):
        return -1
    else:
        return 0

def play_game():
    game = TicTacToe()
    print("Welcome to Tic Tac Toe!")
    print("Here's the current board layout:")
    TicTacToe.print_board_nums()
    print("Let's start the game!")
    print()
    while game.empty_squares():
        if game.num_empty_squares() % 2 == 0:
            square, _ = minimax(None, game.num_empty_squares(), True, game)
            game.make_move(square, 'O')
        else:
            square = None
            while square is None:
                try:
                    square = int(input("Choose a position to place 'X' (0-8): "))
                    if square not in game.available_moves():
                        print("That position is not available. Try again.")
                        square = None
                except ValueError:
                    print("Invalid input. Please enter a number (0-8).")
                    square = None
            game.make_move(square, 'X')
        
        game.print_board()
        print()
        
        if game.current_winner:
            print(f"{game.current_winner} wins!")
            break
    if not game.current_winner:
        print("It's a tie!")

if __name__ == "__main__":
    play_game()
