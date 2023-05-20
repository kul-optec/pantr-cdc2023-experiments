# PANTR: A proximal algorithm with regularized Newton updates for nonconvex constrained optimization

This repository contains a set of benchmarks, including the ones used in the L-CSS/CDC submission of the PANTR method.

### PANTR source code

The source code of PANTR is available on the `develop` branch of the alpaqa repository, at <https://github.com/kul-optec/alpaqa/tree/develop>.

### Instructions (Linux only)

```sh
# Install alpaqa and dependencies, initialize virtual environment
./scripts/get-dependencies.sh
# Generate and compile the benchmark problems and the benchmark driver
./scripts/build-benchmarks.sh
# Activate the virtual environment
. ./.venv/bin/activate
# Run the benchmarks and export the figures
cd new-benchmarks-paper; doit -n$(($(nproc) / 2))
```

---

## Results

### Hanging chain

[`hanging_chain.py`](python/alpaqa_mpc_benchmarks/problems/hanging_chain.py)

Model dynamics from [1].

**Average solver run times for different MPC horizons**  
![Average solver run times and P5/P95 percentiles](images/mpc-hanging_chain-60-avg-runtimes-quantiles-cold-warm.pdf.svg)

**Average solver run times for horizon 60**  
![Solver run times per MPC time step](images/mpc-hanging_chain-60-runtimes-mpc-last-cold-warm.pdf.svg)

### Simplified quadcopter

[`quadcopter.py`](python/alpaqa_mpc_benchmarks/problems/quadcopter.py)

Model dynamics:
$$
    \begin{equation}
        \begin{aligned}
            \dot x &= v \\
            \dot v &= 
            \begin{pmatrix}
            \cos\psi \cos\theta & \cos\psi \sin\theta \sin\phi-\sin\psi \cos\phi & \cos\psi \sin\theta \cos\phi + \sin\psi \sin\phi \\
            \sin\psi \cos\theta & \sin\psi \sin\theta \sin\phi + \cos\psi \cos\phi & \sin\psi \sin\theta \cos\phi - \cos\psi \sin\phi \\
            -\sin\theta & \cos\theta \sin\phi & \cos\theta \cos\phi \\
            \end{pmatrix} \begin{pmatrix} 0 \\ 0 \\ a_t \end{pmatrix} - \begin{pmatrix} 0 \\ 0 \\ g \end{pmatrix} \\
            \begin{pmatrix} \dot \phi \\ \dot \theta \\ \dot \psi \end{pmatrix} &= \omega.
        \end{aligned}
    \end{equation}
$$

**Average solver run times for different MPC horizons**  
![Average solver run times and P5/P95 percentiles](images/mpc-quadcopter-60-avg-runtimes-quantiles-cold-warm.pdf.svg)

**Average solver run times for horizon 60**  
![Solver run times per MPC time step](images/mpc-quadcopter-60-runtimes-mpc-last-cold-warm.pdf.svg)

### Quadcopter

[`realistic_quadcopter.py`](python/alpaqa_mpc_benchmarks/problems/realistic_quadcopter.py)

Model dynamics from [2]: 
$$
    \begin{equation}
        \begin{aligned}
            \dot x &= v \\
            \dot v &= 
            \begin{pmatrix}
            \cos \psi \cos \theta - \sin \phi \sin \psi \sin \theta & -\cos  \phi \sin \psi & \cos  \psi \sin \theta + \cos  \theta \sin \phi \sin \psi \\
            \cos  \theta \sin \psi + \cos  \psi \sin \phi \sin \theta & \cos  \phi \cos  \psi & \sin \psi \sin \theta - \cos  \psi \cos  \theta \sin \phi \\
            -\cos  \phi \sin \theta & \sin \phi & \cos \phi \cos \theta \\
            \end{pmatrix} \begin{pmatrix} 0 \\ 0 \\ a_t \end{pmatrix} - \begin{pmatrix} 0 \\ 0 \\ g \end{pmatrix} \\
            \begin{pmatrix} \dot \phi \\ \dot \theta \\ \dot \psi \end{pmatrix} &= \begin{pmatrix}
            \cos \theta & 0 & -\cos \phi \sin \theta \\
            0 & 1 & \sin \phi \\
            \sin \theta & 0 & \cos \phi \cos \theta \\
            \end{pmatrix} \omega.
        \end{aligned}
    \end{equation}
$$

**Average solver run times for different MPC horizons**  
![Average solver run times and P5/P95 percentiles](images/mpc-realistic_quadcopter-60-avg-runtimes-quantiles-cold-warm.pdf.svg)

**Average solver run times for horizon 60**  
![Solver run times per MPC time step](images/mpc-realistic_quadcopter-60-runtimes-mpc-last-cold-warm.pdf.svg)

---

- [1]&emsp; Wirsching, Leonard & Bock, Hans & Diehl, Moritz. (2006). _Fast NMPC of a chain of masses connected by springs_. Proceedings of the IEEE International Conference on Control Applications. 591 - 596. <https://doi.org/10.1109/CACSD-CCA-ISIC.2006.4776712>
- [2]&emsp; Powers, C., Mellinger, D., Kumar, V. (2015). _Quadrotor Kinematics and Dynamics_. In: Valavanis, K., Vachtsevanos, G. (eds) Handbook of Unmanned Aerial Vehicles. Springer, Dordrecht. <https://doi.org/10.1007/978-90-481-9707-1_71>
