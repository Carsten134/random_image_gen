from torch.fft import fft2, fftshift
from torch import normal, tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def fourier_freq(N):
  return 2 * torch.pi * torch.arange(-(N - 1) // 2, N // 2) / N

def I(x):
  res = abs(fft2(x))**2 / (x.shape[0] * x.shape[1] * torch.var(x))
  return fftshift(res)


def wold_w_q_fast(N, phi_1, phi_2, phi_3):
    wold_weights = torch.full((N, N),0, dtype=torch.float32)
    wold_weights[:, N -1] = phi_3 **torch.arange(N)
    wold_weights[0, [i for i in reversed(range(N))]] = phi_1**torch.arange(N)

    phi = torch.tensor([phi_1, phi_2, phi_3]).reshape(1, 3)
    for i in range(1, N):
        for j in reversed(range(N-1)):
            wold_weights[i, j] = phi @ torch.tensor([wold_weights[i,j+1], wold_weights[i-1, j+1], wold_weights[i-1, j]]).reshape(3, 1)
    
    # position wold weights in the left corner

    res = torch.zeros((2*N -1, 2*N - 1))
    res[(N-1):(2*N-1), 0:N] = wold_weights
    return res


def wold_w_h_fast(N, phi_1, phi_2, phi_3, phi_4):
  phi = torch.tensor([phi_1, phi_2, phi_3, phi_4]).reshape(1,4)

  # diagonal and horizontal weights
  res = torch.full((2*N - 1, 2*N - 1),0, dtype=torch.float32)
  res[[i for i in range(N-1, 2*N-1)], [i for i in range(N-1, 2*N - 1)]] = phi_4 **torch.arange(N)
  res[N-1, [i for i in reversed(range(N))]] = phi_1**torch.arange(N)

  for i in range(N, 2*N-1):
    for j in reversed(range(0, i)):
      if i == 0:
        res = [i, j] = phi @ torch.tensor([res[i,j+1], res[i-1, j+1], res[i-1, j], 0]).reshape(4, 1)
      else:
        res[i, j] = phi @ torch.tensor([res[i,j+1], res[i-1, j+1], res[i-1, j], res[i-1, j-1]]).reshape(4, 1)
  return res

class PlaneSampler(torch.nn.Module):
  def __init__(self, wold_size, phi_1, phi_2, phi_3, phi_4 = 0):
    super().__init__()
    # constructing wold weights 
    K_buffer = wold_w_q_fast(wold_size, phi_1,  phi_2, phi_3) if phi_4 == 0 else wold_w_h_fast(wold_size, phi_1, phi_2, phi_3, phi_4)
    
    # register K in the buffer, such that it can be moved to a CUDA device
    self.register_buffer("K", K_buffer.reshape(1,1,*K_buffer.shape))
    self.padding = K_buffer.shape[0] // 2

  def forward(self, N, M, T = 1):
    N_tilde = N + 2 * self.padding
    M_tilde = M + 2 * self.padding
    eps = normal(0, 1, size = (T, N_tilde, M_tilde))
    eps = eps.reshape(T,1, N_tilde, M_tilde)
    return F.conv2d(eps, self.K)
  
  def plot(self, N = 100):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    x = self.forward(N, N)[0][0]

    fig.dpi = 300
    ax1.imshow(x)
    ax1.set_title("Sample")
    ax2.imshow(I(x))
    ax2.set_title("Spectrum")
    ax3.imshow(self.K[0][0])
    ax3.set_title("Wold weights")

    return fig