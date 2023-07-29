import chess
import random
import math
import evaluate

class Matsya:

    def __init__(self, board):
        self.board = board
        self.depth = 3
    
    def get_current_position(self):
        return self.board.fen()
    
    def make_move(self, move):
        self.board.push(move)

    def get_best_move(self):
        best_move = chess.Move.null()
        depth = 2

        if self.board.turn:
            best_move, _ = self.search(depth, self.board, True)
        else:
            best_move, _ = self.search(depth, self.board, False)

        return best_move

    def search(self, depth, board, maximize):

        legals = board.legal_moves
        bestMove = None
        bestValue = -9999
        if(not maximize):
            bestValue = 9999
        for move in legals:
            board.push(move)
            value = self.alphaBeta(board, depth-1, -10000, 10000, (not maximize))
            board.pop()
            if maximize:
                if value > bestValue:
                    bestValue = value
                    bestMove = move
            else:
                if value < bestValue:
                    bestValue = value
                    bestMove = move
        return (bestMove, bestValue)

    def alphaBeta(self, board, depth, alpha, beta, maximize):
        if(board.is_checkmate()):
            if(board.turn == chess.WHITE):
                return -10000
            else:
                return 10000
        if depth == 0:
            val = self.evaluate(board)
            if val is not None:
                return val
            else:
                return 0
        legals = board.legal_moves
        if(maximize):
            bestVal = -9999
            for move in legals:
                board.push(move)
                bestVal = max(bestVal, self.alphaBeta(board, depth-1, alpha, beta, (not maximize)))
                board.pop()
                alpha = max(alpha, bestVal)
                if alpha >= beta:
                    return bestVal
            return bestVal
        else:
            bestVal = 9999
            for move in legals:
                board.push(move)
                bestVal = min(bestVal, self.alphaBeta(board, depth - 1, alpha, beta, (not maximize)))
                board.pop()
                beta = min(beta, bestVal)
                if beta <= alpha:
                    return bestVal
            return bestVal


    def evaluate(self, board):
        # first check for mate or draw
        if board.is_checkmate():
            if board.turn:
                return -math.inf
            else:
                return math.inf
        
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        value = evaluate.evaluate_board(board)
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
                self.handle_go_command()
            elif command[0] == "quit":
                exit()
    
    def uci_protocol(self):
        print("id name Matsya Engine")
        print("id author Paritosh Dahiya")
        print("uciok")
    
    def handle_position_command(self, command):
        if command[1] == "startpos":
            self.board.set_fen(chess.STARTING_FEN)
            command = command[1:]

            moves = command[2:]
            for move in moves:
                self.board.push_uci(move)

    
    def handle_go_command(self):
        best_move = self.get_best_move()
        if best_move is None or best_move == chess.Move.null():
            best_move = random.choice(list(self.board.legal_moves))
        self.board.push(best_move)
        print("bestmove " + str(best_move))
        

if __name__ == "__main__":
    engine = Matsya(chess.Board())
    engine.uci_loop()