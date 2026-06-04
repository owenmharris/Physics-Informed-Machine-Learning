import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class PINN(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    
    def forward(self, t):
        return self.net(t)
    

m = 1.0
k = 1.0
mu = 0.1


def physics_loss(model, t_physics):

    t_physics = t_physics.requires_grad_(True)
    x = model(t_physics)

    dx_dt = torch.autograd.grad(x, t_physics,
                                grad_outputs = torch.ones_like(x),
                                create_graph = True) [0]

    d2x_dt2 = torch.autograd.grad(dx_dt, t_physics,
                                  grad_outputs = torch.ones_like(dx_dt),
                                  create_graph = True) [0]

    residual = m * d2x_dt2 + mu * dx_dt + k * x
    return torch.mean(residual**2)


model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

t_physics = torch.linspace(0, 10, 100).view(-1, 1)

t_data = torch.tensor([[0.0]]).view(-1, 1)
x_data = torch.tensor([[1.0]]).view(-1, 1)

for epoch in range(5000):
    optimizer.zero_grad()

    x_pred = model(t_data)
    data_loss = torch.mean((x_pred - x_data)**2)

    phys_loss = physics_loss(model, t_physics)

    loss = data_loss + 1000 * phys_loss

    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")


t_plot = torch.linspace(0, 10, 200).view(-1, 1)
x_pred = model(t_plot).detach().numpy()

plt.plot(t_plot.numpy(), x_pred, label="PINN Prediction")
plt.xlabel("Time")
plt.ylabel("Position")
plt.legend()
plt.title("Simple Harmonic Oscillator with PINN")
plt.grid(True)
plt.show()