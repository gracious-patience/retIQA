import torch as th
from .nn import timestep_embedding

class SinFuser(th.nn.Module):
    def __init__(self, k: int, before_initial_conv: bool):
        super().__init__()
        self.k = k
        self.before_initial_conv = before_initial_conv 

        # modules
        self.initial_conv = th.nn.Conv2d(3*(k+1), 3, kernel_size=(1, 1), bias=0)
        if before_initial_conv:
            self.post_process = th.nn.Conv2d(
                3*k, 3*k, (3,3), padding=(1,1)
            )
        else:
            self.post_process = th.nn.Conv2d(
                3, 3, (3,3), padding=(1,1)
            )
        self.nonlinearity = th.nn.SiLU()
     
    def forward(self, x, y):
        if self.before_initial_conv:
            embeds = timestep_embedding(timesteps=y, dim=3*224*224).reshape([
                x.shape[0], self.k * 3, 224, 224
            ])
            embeds = self.post_process(embeds)
            embeds = self.nonlinearity(embeds)
            embeds = th.cat([ th.zeros([x.shape[0], 3, 224, 224], device=x.device), embeds ], dim=1)
            x = x + embeds
            res = self.initial_conv(x)
            return self.nonlinearity(res)
        else:
            embeds = timestep_embedding(timesteps=y, dim=3*224*224//self.k).reshape([
                x.shape[0], 3, 224, 224
            ])
            embeds = self.post_process(embeds)
            embeds = self.nonlinearity(embeds)
            x = self.initial_conv(x)
            res = x + embeds
            return self.nonlinearity(res)
