import torch
import torch.nn as nn

def init_weights(layer, mean=0.0, std=0.02):
    if isinstance(layer,(nn.Conv3d,nn.ConvTranspose2d)):
        nn.init.normal_(layer.weight,mean,std)
    elif isinstance(layer,(nn.Linear,nn.BatchNorm2d)):
        nn.init.normal_(layer.weight,mean,std)
        nn.init.constant_(layer.bias,0)

class Reshape(nn.Module):
    def __init__(self,shape):
        super().__init__()
        self.shape=shape
    def forward(self,x):
        return x.view(x.size(0),*self.shape)
 
    
class TemporalNetwork(nn.Module):
    def __init__(
        self,
        z_dimension=32,
        hid_channels=1024,
        n_bars=2):
        super().__init__()
        self.n_bars = n_bars
        self.net = nn.Sequential(
            # input shape: (batch_size, z_dimension)
            Reshape(shape=[z_dimension, 1, 1]),
            # output shape: (batch_size, z_dimension, 1, 1)
            nn.ConvTranspose2d(
                z_dimension,
                hid_channels,
                kernel_size=(2, 1),
                stride=(1, 1),
                padding=0,
            ),
            nn.BatchNorm2d(hid_channels),
            nn.ReLU(inplace=True),
            # output shape: (batch_size, hid_channels, 2, 1)
            nn.ConvTranspose2d(
                hid_channels,
                z_dimension,
                kernel_size=(self.n_bars - 1, 1),
                stride=(1, 1),
                padding=0,
            ),
            nn.BatchNorm2d(z_dimension),
            nn.ReLU(inplace=True),
            # output shape: (batch_size, z_dimension, 1, 1)
            Reshape(shape=[z_dimension, self.n_bars]),
        )

    def forward(self, x):
        return self.net(x)
    
class MuseCritic(nn.Module):
    def __init__(self,hid_channels=128,hid_features=1024,
        out_features=1,n_tracks=4,n_bars=2,n_steps_per_bar=16,
        n_pitches=84):
        super().__init__()
        self.n_tracks = n_tracks
        self.n_bars = n_bars
        self.n_steps_per_bar = n_steps_per_bar
        self.n_pitches = n_pitches
        in_features = 4 * hid_channels if n_bars == 2 else 12 * hid_channels
        self.seq = nn.Sequential(
            nn.Conv3d(self.n_tracks, hid_channels, 
                      (2, 1, 1), (1, 1, 1), padding=0),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv3d(hid_channels, hid_channels, 
              (self.n_bars - 1, 1, 1), (1, 1, 1), padding=0),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv3d(hid_channels, hid_channels, 
                      (1, 1, 12), (1, 1, 12), padding=0),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv3d(hid_channels, hid_channels, 
                      (1, 1, 7), (1, 1, 7), padding=0),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv3d(hid_channels, hid_channels, 
                      (1, 2, 1), (1, 2, 1), padding=0),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv3d(hid_channels, hid_channels, 
                      (1, 2, 1), (1, 2, 1), padding=0),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv3d(hid_channels, 2 * hid_channels, 
                      (1, 4, 1), (1, 2, 1), padding=(0, 1, 0)),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv3d(2 * hid_channels, 4 * hid_channels, 
                      (1, 3, 1), (1, 2, 1), padding=(0, 1, 0)),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Flatten(),
            nn.Linear(in_features, hid_features),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Linear(hid_features, out_features))
    def forward(self, x):  
        return self.seq(x)
     
   
class BarGenerator(nn.Module):
    def __init__(self,z_dimension=32,hid_features=1024,hid_channels=512,
        out_channels=1,n_steps_per_bar=16,n_pitches=84):
        super().__init__()
        self.n_steps_per_bar = n_steps_per_bar
        self.n_pitches = n_pitches
        self.net = nn.Sequential(
            nn.Linear(4 * z_dimension, hid_features),
            nn.BatchNorm1d(hid_features),
            nn.ReLU(inplace=True),
            Reshape(shape=[hid_channels,hid_features//hid_channels,1]),
            nn.ConvTranspose2d(hid_channels,hid_channels,
               kernel_size=(2, 1),stride=(2, 1),padding=0),
            nn.BatchNorm2d(hid_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hid_channels,hid_channels // 2,
                kernel_size=(2, 1),stride=(2, 1),padding=0),
            nn.BatchNorm2d(hid_channels // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hid_channels // 2,hid_channels // 2,
                kernel_size=(2, 1),stride=(2, 1),padding=0),
            nn.BatchNorm2d(hid_channels // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hid_channels // 2,hid_channels // 2,
                kernel_size=(1, 7),stride=(1, 7),padding=0),
            nn.BatchNorm2d(hid_channels // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hid_channels // 2,out_channels,
                kernel_size=(1, 12),stride=(1, 12),padding=0),
            Reshape([1, 1, self.n_steps_per_bar, self.n_pitches]))
    def forward(self, x):
        return self.net(x)

    

def loss_fn(pred,target):
    return -torch.mean(pred*target)

class GradientPenalty(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inputs, outputs):
        grad = torch.autograd.grad(
            inputs=inputs,
            outputs=outputs,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True,
            retain_graph=True,
        )[0]
        grad_=torch.norm(grad.view(grad.size(0),-1),p=2,dim=1)
        penalty = torch.mean((1. - grad_) ** 2)
        return penalty



class MuseGenerator(nn.Module):
    def __init__(self,z_dimension=32,hid_channels=1024,
        hid_features=1024,out_channels=1,n_tracks=4,
        n_bars=2,n_steps_per_bar=16,n_pitches=84):
        super().__init__()
        self.n_tracks = n_tracks
        self.n_bars = n_bars
        self.n_steps_per_bar = n_steps_per_bar
        self.n_pitches = n_pitches
        self.chords_network=TemporalNetwork(z_dimension, 
                            hid_channels, n_bars=n_bars)
        self.melody_networks = nn.ModuleDict({})
        for n in range(self.n_tracks):
            self.melody_networks.add_module(
                "melodygen_" + str(n),
                TemporalNetwork(z_dimension, 
                 hid_channels, n_bars=n_bars))
        self.bar_generators = nn.ModuleDict({})
        for n in range(self.n_tracks):
            self.bar_generators.add_module(
                "bargen_" + str(n),BarGenerator(z_dimension,
            hid_features,hid_channels // 2,out_channels,
            n_steps_per_bar=n_steps_per_bar,n_pitches=n_pitches))
    def forward(self,chords,style,melody,groove):
        chord_outs = self.chords_network(chords)
        bar_outs = []
        for bar in range(self.n_bars):
            track_outs = []
            chord_out = chord_outs[:, :, bar]
            style_out = style
            for track in range(self.n_tracks):
                melody_in = melody[:, track, :]
                melody_out = self.melody_networks["melodygen_"\
                          + str(track)](melody_in)[:, :, bar]
                groove_out = groove[:, track, :]
                z = torch.cat([chord_out, style_out, melody_out,\
                               groove_out], dim=1)
                track_outs.append(self.bar_generators["bargen_"\
                                          + str(track)](z))
            track_out = torch.cat(track_outs, dim=1)
            bar_outs.append(track_out)
        out = torch.cat(bar_outs, dim=2)
        return out
    
    

































 