# Financial Derivative Calculator with Risk Analysis

This Python repository offers a comprehensive suite of tools to calculate derivative prices and risk metrics, and conduct Monte Carlo simulations for financial derivatives. The application is built using Streamlit for a user-friendly interface.

## Features

### Derivative Pricing Models

#### Black-Scholes Model
The `calculate_black_scholes` function computes option prices using the Black-Scholes pricing model. Users can input various parameters such as underlying asset price, strike price, volatility, interest rate, time to expiry, dividend yield, and option type (Call or Put).

#### Binomial Tree Model
The `calculate_binomial_tree` function provides option pricing via the Binomial Tree model. Similar to the Black-Scholes model, users can specify parameters to derive option prices.

### Additional Functions

- `calculate_futures_price`: Computes the futures price given the spot price, risk-free rate, and time to maturity.
- `generate_price_paths`: Generates simulated price paths using Monte Carlo simulations based on Geometric Brownian Motion (GBM).
- `risk_metrics_calc`: Calculates risk metrics such as Delta, Gamma, Vega, and Theta for options.

### Usage

The application utilizes Streamlit for a user interface. Users can interact with the sidebar to select the derivative type (Option, Futures, or Swaps) and input the relevant parameters. The calculated prices and risk metrics are displayed accordingly.

### Monte Carlo Simulation

The application supports Monte Carlo simulations for selected derivatives. Users can specify the number of simulated price paths, time steps, drift, volatility, and time horizon for the simulation.

## How to Use

1. **Installation:**
    - Clone the repository.
    - Install the required dependencies: `streamlit`, `numpy`, `scipy`, `matplotlib`.

2. **Running the Application:**
    - Execute the main Python file: `python main.py`.
    - Access the application via the provided URL in the terminal.

## Dependencies

- Streamlit
- NumPy
- SciPy
- Matplotlib


## License

This project is licensed under the [MIT License].
