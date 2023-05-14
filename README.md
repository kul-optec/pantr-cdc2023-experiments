# PANTR: A proximal algorithm with regularized Newton updates for nonconvex constrained optimization

This repository contains a set of benchmarks, including the ones used in the L-CSS/CDC submission of the PANTR method.

### PANTR source code

The source code of PANTR is available on the `pantr` branch of the alpaqa repository, at <https://github.com/kul-optec/alpaqa/tree/pantr>.

### Instructions (Linux only)

```sh
# Install alpaqa and dependencies, initialize virtual environment
./scripts/get-dependencies.sh
# Generate and compile the benchmark problems and the benchmark driver
./scripts/build-benchmarks.sh
# Run the benchmarks and export the figures
cd benchmarks-paper; make -j$(($(nproc) / 2))
```

---

## Results

### Hanging chain

🚧 (in progress) 🚧

### Simplified quadcopter

🚧 (in progress) 🚧

### Quadcopter

🚧 (in progress) 🚧
