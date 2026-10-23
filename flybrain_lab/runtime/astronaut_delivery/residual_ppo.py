"""Small on-policy residual PPO. Never grants permission to move.

The stochastic action is a latent Normal vector; tanh maps it to a bounded
proposal. PPO likelihood ratios use the latent action (the fixed bijection's
Jacobian cancels). IK projection and the safety shield belong to the environment.
"""
import numpy as np
import torch
from torch import nn

OBSERVATIONS=107
ACTION_RADIUS=.02


class ResidualPPO:
    def __init__(self,seed=41,training=True):
        torch.manual_seed(seed);torch.set_num_threads(1)
        self.training=training
        self.actor=nn.Sequential(nn.Linear(OBSERVATIONS,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,16))
        self.critic=nn.Sequential(nn.Linear(OBSERVATIONS,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,1))
        nn.init.zeros_(self.actor[-1].weight);nn.init.zeros_(self.actor[-1].bias)
        self.log_std=nn.Parameter(torch.full((16,),-.7))
        self.parameters=list(self.actor.parameters())+list(self.critic.parameters())+[self.log_std]
        self.optimizer=torch.optim.Adam(self.parameters,lr=3e-4)
        self.buffer=[];self.updates=[]

    def sample(self,observation):
        x=torch.as_tensor(observation,dtype=torch.float32)
        if x.shape!=(OBSERVATIONS,) or not torch.isfinite(x).all():raise ValueError('invalid deployable observation')
        with torch.no_grad():
            d=torch.distributions.Normal(self.actor(x),self.log_std.exp())
            z=d.sample() if self.training else d.mean
            row=dict(observation=x.numpy().tolist(),latent=z.numpy().tolist(),
                log_probability=float(d.log_prob(z).sum()),value=float(self.critic(x).squeeze()))
        return ACTION_RADIUS*np.tanh(z.numpy()),row

    def finish(self,row,reward,next_observation,terminal=False):
        if not self.training:return
        with torch.no_grad():bootstrap=0. if terminal else float(self.critic(torch.as_tensor(next_observation,dtype=torch.float32)).squeeze())
        self.buffer.append(dict(row,reward=float(reward),bootstrap=bootstrap,terminal=terminal))
        if len(self.buffer)>=16 or terminal:self.update()

    def update(self):
        if not self.buffer:return
        b=self.buffer;adv=[];gae=0.
        for row in reversed(b):
            delta=row['reward']+.99*row['bootstrap']-row['value']
            gae=delta+.99*.95*(not row['terminal'])*gae;adv.append(gae)
        a=torch.tensor(adv[::-1],dtype=torch.float32)
        returns=a+torch.tensor([r['value'] for r in b]);a=(a-a.mean())/(a.std(unbiased=False)+1e-6)
        x=torch.tensor([r['observation'] for r in b]);z=torch.tensor([r['latent'] for r in b])
        old=torch.tensor([r['log_probability'] for r in b])
        for _ in range(4):
            d=torch.distributions.Normal(self.actor(x),self.log_std.exp())
            ratio=(d.log_prob(z).sum(-1)-old).exp()
            policy=-torch.minimum(ratio*a,ratio.clamp(.8,1.2)*a).mean()
            value=.5*(self.critic(x).squeeze(-1)-returns).square().mean()
            loss=policy+value-.01*d.entropy().sum(-1).mean()
            if not torch.isfinite(loss):raise RuntimeError('nonfinite PPO loss')
            self.optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.parameters,.5);self.optimizer.step()
        self.updates.append(dict(samples=len(b),mean_reward=float(np.mean([r['reward'] for r in b])),loss=float(loss.detach())))
        self.buffer=[]

    def save(self,path):
        if self.buffer:raise RuntimeError('checkpoint requires flushed on-policy rollout')
        torch.save(dict(actor=self.actor.state_dict(),critic=self.critic.state_dict(),log_std=self.log_std.detach(),
            observation_size=OBSERVATIONS,action_radius=ACTION_RADIUS,optimizer=self.optimizer.state_dict(),
            updates=self.updates,rng_state=torch.get_rng_state(),checkpoint_version=2),path)

    def load(self,path,resume=False):
        data=torch.load(path,map_location='cpu',weights_only=True)
        if data['observation_size']!=OBSERVATIONS or data['action_radius']!=ACTION_RADIUS:raise ValueError('policy schema mismatch')
        self.actor.load_state_dict(data['actor']);self.critic.load_state_dict(data['critic'])
        with torch.no_grad():self.log_std.copy_(data['log_std'])
        self.resume_kind='weights_only'
        if resume and 'optimizer' in data:
            self.optimizer.load_state_dict(data['optimizer'])
            self.updates=list(data['updates']);torch.set_rng_state(data['rng_state'])
            self.resume_kind='optimizer_and_rng'
