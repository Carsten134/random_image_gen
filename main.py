import streamlit as st
import pandas as pd
from image_sim.simulation import PlaneSampler

st.write("""
         # Random Image generator
         Choose your parameters and start generating!
         """)

phi_1 = st.slider("Select $\phi_1$:", 0.0, 1.0, .3)
phi_2 = st.slider("Select $\phi_2$:", 0.0, 1.0, .3)
phi_3 = st.slider("Select $\phi_3$:", 0.0, 1.0, .3)

st.write(f"""
  Chosen process:
$$
\\begin{{align*}}
      X_{{ij}} &= \phi_1 X_{{i,j-1}}+\phi_2 X_{{i-1, j-1}} +\phi_3 X_{{i-1,j}} + \epsilon_{{ij}} \\\\
         &= {phi_1}X_{{i-1,j}} + {phi_2} X_{{i-1, j-1}} + {phi_3}X_{{i-1,j}}   
\\end{{align*}}
$$
""")

res = st.slider("Resolution",25, 300, 100)

if st.button("Generate", type = "primary"):
  sampler = PlaneSampler(60, phi_1, phi_2, phi_3)
  fig = sampler.plot(res)
  st.pyplot(fig, dpi = 200)