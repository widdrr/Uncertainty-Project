## Prerequisites

- Docker installed on your system

## How to run

1. Build the Docker image:
```sh
docker build -t quantum-bounds .
```

2. Run the container to generate plots:
```sh
docker run --rm -v "$(pwd)/plots:/app/plots" quantum-bounds
```

The generated plots will be available in the `plots` directory:
- `lower_bound_plots.png`: Line plots comparing different formulas
- `lower_bound_heatmaps.png`: Heatmaps of all formulas
- `f4_differences.png`: Differences with Formula 4
- `f12_differences.png`: Differences with Formula 12
- `f13_differences.png`: Differences with Formula 13
- `f14_differences.png`: Differences with Formula 14
- `optimal_differences.png`: Differences with optimal bounds