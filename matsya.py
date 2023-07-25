import chess
import random
import math
import neural_net
import torch

class Matsya:

    def __init__(self, board):
        self.board = board
        self.depth = 3
        self.time_control = 1000
        self.eval_model = neural_net.EvalNetwork()
        self.eval_model.load()
    
    def get_current_position(self):
        return self.board.fen()
    
    def make_move(self, move):
        self.board.push(move)

    def get_best_move(self):
        # legal_moves = self.board.legal_moves
        # best_move = None
        # best_score = -math.inf

        best_move = self.search()
        return best_move

    def search(self, depth):
        best_move = None
        best_score = -math.inf
        alpa = -math.inf
        beta = math.inf
        for move in self.board.legal_moves():
            pass

    
    def alpha_beta(self, board, depth, alpha, beta, maximizing_player):
        if depth == 0 or board.is_game_over():
            return self.evaluate(board.fen())
        
        

    def evaluate(self, board):
        # first check for mate or draw
        if board.is_checkmate():
            if board.turn:
                return -math.inf
            else:
                return math.inf
        
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        value = self.eval_model.inference(board.fen())
        return value



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