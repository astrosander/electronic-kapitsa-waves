# Electronic Kapitsa Waves  
**Current-driven hydrodynamic instabilities in electron fluids**  

## 📖 Overview  
This project explores **Kapitsa-like instabilities**—waves that only exist in moving media—in **electron fluids** such as high-mobility graphene. Inspired by P. Kapitsa’s classic roll-wave experiments in viscous liquids, we investigate their **electronic analog** using **Fermi liquid hydrodynamics** with momentum dissipation.  

We model, simulate, and analyze conditions where these unusual waves can arise, linking theory to recent experimental findings in graphene transport.  

---

## ✨ Features  
- **Hydrodynamic Model** – Implements Fermi liquid pressure, momentum relaxation, and current-driven terms.  
- **Instability Analysis** – Derives and solves the dispersion relation for electronic Kapitsa waves.  
- **Phase Diagram** – Identifies stable and unstable regimes as a function of flow velocity, carrier density, and dissipation rate.  
- **Graphene Context** – Connects predictions to experimentally accessible parameters in mono-, multi-layer, and moiré graphene systems.  
- **Visualization** – Plots wave profiles, growth rates, and critical thresholds.  

---

## 🚀 Usage Examples  
Run the **phase diagram generation**:  
```bash
python phase_diagram.py --velocity_range 0.01 1.0 --density_range 1e11 1e13
````

Run the **dispersion analysis** for given parameters:

```bash
python dispersion_analysis.py --v 0.3 --n 5e12 --gamma 1e12
```

Or explore interactive **Jupyter notebooks** for quick experiments.

---

## 🔬 Theory Background

Kapitsa waves are a **flow-driven instability** observed in viscous films on inclined planes, first studied by **Pyotr Kapitsa**. In electronic systems, a similar effect may occur when electrons behave as a **viscous fluid** under strong driving currents, even in the presence of strong momentum damping.

The governing hydrodynamic equations are:

$$
\partial_t n + \nabla \cdot j = 0, \quad
\partial_t p + \nabla \Pi + \gamma p = e n E
$$

where $n$ is particle density, $p$ is momentum density, $\gamma$ is momentum dissipation, and $\Pi$ includes Fermi pressure.

---

## 📚 References

1. P.L. Kapitza, *Wave flow of thin layers of a viscous fluid*, Zh. Eksp. Teor. Fiz., 18, 3–28 (1948).
2. Berdyugin et al., *Out-of-equilibrium criticality in graphene*, Science 375, 430–433 (2022).
3. Balmforth & Mandre, *Dynamics of roll waves*, J. Fluid Mech. (2004).

---

## 🤝 Contributing

Contributions, issues, and pull requests are welcome!

---

## 📜 License

MIT License – see `LICENSE` file.
If you want, I can now make you a **minimal runnable code example** for `dispersion_analysis.py` so your repo launches with ready-to-run physics plots. That will make it instantly attractive on GitHub.
