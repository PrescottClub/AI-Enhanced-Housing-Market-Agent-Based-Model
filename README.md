# Urban Housing Market and Gentrification Simulation

## Overview

This project implements an agent-based model to simulate urban housing market dynamics and gentrification processes. It explores how interactions between residents, investors, and businesses shape property values, income inequality, and neighborhood diversity in different market sizes.

## Features

- Agent-based modeling of residents, investors, properties, and businesses
- Simulation of property value changes, resident satisfaction, and investment decisions
- Analysis of gentrification indicators including Gini coefficient and neighborhood diversity
- Parameter studies to explore the impact of market size and investor numbers
- Visualization of simulation results using matplotlib

## Requirements

- Python 3.7+
- NumPy
- Matplotlib
- Pandas
- Mesa (Agent-based modeling framework)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/urban-housing-simulation.git
   ```
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the main simulation:
   ```
   python run_simulation.py
   ```
2. To conduct a parameter study:
   ```
   python parameter_study.py
   ```
3. View results in the `output` directory

## Model Description

The model consists of four main agent types:
- Residents: Make housing decisions based on income and satisfaction
- Investors: Buy and sell properties based on expected returns
- Properties: Change in value based on neighborhood characteristics
- Businesses: Establish in areas based on local income levels

The simulation runs for a specified number of steps, during which agents interact and make decisions, leading to emergent patterns of gentrification and urban change.

## Results

The model produces several key outputs:
- Time series of average property values, Gini coefficient, and neighborhood diversity
- Distributions of resident incomes, property values, and satisfaction levels
- Visualizations of these metrics under different parameter settings

## Contributing

Contributions to improve the model or extend its capabilities are welcome. Please feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

This project was inspired by and builds upon the work of:
- Benenson & Torrens (2004) on geosimulation
- Schelling (1971) on segregation dynamics
- Recent work on agent-based modeling of urban systems (see References in the main report)

## Contact

For any queries regarding this project, please open an issue on GitHub or contact [Your Name] at [your.email@example.com].