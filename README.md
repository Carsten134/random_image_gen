## Random image generato
For my Master's thesis I am investigating two dimensional stationary processes. One such process is the **quarterplane process**:
$$
X_{ij} = \begin{pmatrix}
  \phi_1 & 1 \\
  \phi_2 & \phi_3
\end{pmatrix} \odot \begin{pmatrix}
  X_{i,j-1}& \varepsilon_{ij} \\
  X_{i-1,j-1} & X_{i-1, j}
\end{pmatrix}
$$

With $\varepsilon_{ij} \sim WWN(0,1)$ and $\odot$ being the elementwise product.

I personally had a lot of fun playing around with the samplegeneration of this process. So i built this small webapp in streamlit, such that you can too :smile:.