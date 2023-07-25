import torch.nn as nn
import torch

class EvalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(65, 1024)
        self.hidden_layer = nn.Linear(1024, 512)
        self.output_layer = nn.Linear(512, 1)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = nn.functional.relu(x)
        x = self.hidden_layer(x)
        x = nn.functional.relu(x)
        x = self.output_layer(x)
        return x
    
    def load(self):
        self.load_state_dict(torch.load('model/nn_1024_512.pt'))

    def inference(self, board):

        piece_val_map = {'p':-1,'k':-600,'b':-3,'r':-5,'q':-9,'n':-3,'P':1,'K':600,'B':3,'R':5,'Q':9,'N':3}

        input_data = []
        for char in board.split(" ")[0]:
            if char.isdigit():
                for _ in range(int(char)):
                    input_data.append(0)
            else:
                if char == '/':
                    continue
                input_data.append(piece_val_map[char])

        if board.split(" ")[1] == "w":
            input_data.append(1)
        else:
            input_data.append(0)

        input_data = torch.tensor(input_data, dtype=torch.float32)

        score = self(input_data).tolist()[0]
        print(score)

# model = EvalNetwork()
# model.load()
# model.inference('r2qkbr1/pb1nn3/1ppp3p/8/3P1p2/2PB1N1P/PPQN1PP1/2K1R2R w q - 2 15')