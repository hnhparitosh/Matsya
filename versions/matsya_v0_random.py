import chess
import random

class Matsya:

    def __init__(self, board):
        self.board = board
        self.search_depth = 3
        self.time_control = 1000
    
    def get_current_position(self):
        return self.board.fen()
    
    def make_move(self, move):
        self.board.push(move)

    def get_best_move(self):
        moves = list(self.board.legal_moves)
        # print(moves)
        best_move = random.choice(moves)
        return best_move

    def get_evaluation_of_position(self):
        return chess.engine.AlphaBetaPruning(self.board, depth=self.search_depth, time_limit=self.time_control).evaluation()

    def set_search_depth(self, depth):
        self.search_depth = depth

    def set_time_control(self, time_control):
        self.time_control = time_control
    
    def uci_loop(self):
        while True:
            command = input().split()
            if command[0] == "uci":
                self.uci_protocol()
            elif command[0] == "isready":
                print("readyok")
            elif command[0] == "ucinewgame":
                self.board = chess.Board()
            elif command[0] == "position":
                self.handle_position_command(command)
            elif command[0] == "go":
                self.handle_go_command(command)
            elif command[0] == "quit":
                exit()
    
    def uci_protocol(self):
        print("id name Matsya_v0_Random")
        print("id author Paritosh Dahiya")
        print("uciok")
    
    def handle_position_command(self, command):
        if command[1] == "startpos":
            self.board.set_fen(chess.STARTING_FEN)
            command = command[1:]

            moves = command[2:]
            for move in moves:
                self.board.push_uci(move)

    
    def handle_go_command(self, command):
        best_move = self.get_best_move()
        self.board.push(best_move)
        print("bestmove " + str(best_move))
        

if __name__ == "__main__":
    engine = Matsya(chess.Board())
    engine.uci_loop()