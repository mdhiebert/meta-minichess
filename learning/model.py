import torch.nn as nn

class MiniChessModel(nn.Module):
    def __init__(self):
        super(MiniChessModel, self).__init__()
        self.conv1 = nn.Conv2d(7, 256, kernel_size=5, stride=1, padding=2) # (5,5) -> (5,5)
        self.batch1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2) # (5, 5) -> (5,5)
        self.batch2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0) # (5,5) -> (3,3)
        self.batch3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0) # (3,3) -> (1,1)

        self.fc1 = nn.Linear(512, 256)
        self.piece_choice = nn.Linear(256, 10)
        self.type_choice = nn.Linear(256, 15)
        self.mag_choice = nn.Linear(256, 8)

        self.softmax = nn.Softmax()

    def forward(self, x):

        # formatting
        x = x.permute([2, 0 ,1])
        x = x.unsqueeze(0).float()


        x = self.conv1(x)
        x = self.batch1(x)
        
        x = self.conv2(x)
        x = self.batch2(x)

        x = self.conv3(x)
        x = self.batch3(x)

        x = self.conv4(x)

        x = x.squeeze().unsqueeze(0)

        x = self.fc1(x)
        
        piece = self.piece_choice(x)
        piece = self.softmax(piece).squeeze()

        _type = self.type_choice(x)
        _type = self.softmax(_type).squeeze()

        magnitude = self.mag_choice(x)
        magnitude = self.softmax(magnitude).squeeze()

        return piece, _type, magnitude