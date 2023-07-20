import chess

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
        best_move = chess.engine.AlphaBetaPruning(self.board, depth=self.search_depth, time_limit=self.time_control).best_move()
        return best_move

    def get_evaluation_of_position(self):
        return chess.engine.AlphaBetaPruning(self.board, depth=self.search_depth, time_limit=self.time_control).evaluation()

    def set_search_depth(self, depth):
        self.search_depth = depth

    def set_time_control(self, time_control):
        self.time_control = time_control
    
    def uci_loop(self):
        while True:
            command = input()
            if command == "uci":
                self.uci_protocol()
            elif command.startswith("isready"):
                print("readyok")
            elif command.startswith("position"):
                self.handle_position_command(command)
            elif command.startswith("go"):
                self.handle_go_command(command)
            elif command == "quit":
                break
            else:
                print("Unknown command: " + command)
    
    def uci_protocol(self):
        print("id name Matsya Engine")
        print("id author Paritosh Dahiya (hnhparitosh)")
        print("uciok")
    
    def handle_position_command(self, command):
        parts = command.split(" ")
        if len(parts) != 5:
            print("Invalid position command: " + command)
            return

        fen = parts[1]
        start_pos = parts[2]
        end_pos = parts[3]
        moves = parts[4]

        self.board.set_fen(fen)
        self.board.set_position(start_pos)
        self.board.push_san(moves)
    
    def handle_go_command(self, command):
        parts = command.split(" ")
        if len(parts) not in (2, 3):
            print("Invalid go command: " + command)
            return

        depth = int(parts[1]) if len(parts) == 3 else self.search_depth
        time_control = int(parts[2]) if len(parts) == 3 else self.time_control

        best_move = chess.engine.AlphaBetaPruning(self.board, depth=depth, time_limit=time_control).best_move()
        print(best_move)

if __name__ == "__main__":
    engine = Matsya(chess.Board())
    engine.uci_loop()