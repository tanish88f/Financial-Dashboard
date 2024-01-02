import streamlit as st
import numpy as np
from scipy.stats import norm
import numpy as np
from datetime import date
import matplotlib.pyplot as plt


def calculate_black_scholes(underlying_price, strike_price, volatility, interest_rate, time_to_expiry, dividend_yield, option_type):
        d1 = (np.log(underlying_price / strike_price) + (interest_rate - dividend_yield + (volatility ** 2) / 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
        d2 = d1 - volatility * np.sqrt(time_to_expiry)
        if option_type == 'Call':
            option_price = underlying_price * np.exp(-dividend_yield * time_to_expiry) * norm.cdf(d1) - strike_price * np.exp(-interest_rate * time_to_expiry) * norm.cdf(d2)
        elif option_type == 'Put':
            option_price = strike_price * np.exp(-interest_rate * time_to_expiry) * norm.cdf(-d2) - underlying_price * np.exp(-dividend_yield * time_to_expiry) * norm.cdf(-d1)
        return option_price

def calculate_binomial_tree(underlying_price, strike_price, volatility, interest_rate, time_to_expiry, dividend_yield, option_type, steps):
    time_to_expiry /= 52.143*7  # Convert from years to weeks
    dt = time_to_expiry / steps
    u = np.exp(volatility * np.sqrt(dt))
    d = 1 / u
    r = np.exp(interest_rate * dt) - 1
    p = (np.exp((interest_rate - dividend_yield) * dt) - d) / (u - d)
    q = 1 - p
    price_tree = np.zeros([steps + 1, steps + 1])
    for i in range(steps + 1):
        for j in range(i + 1):
            price_tree[j, i] = underlying_price * (d ** j) * (u ** (i - j))
    option_tree = np.zeros([steps + 1, steps + 1])
    if option_type == 'Call':
        option_tree[:, steps] = np.maximum(np.zeros(steps + 1), price_tree[:, steps] - strike_price)
    elif option_type == 'Put':
        option_tree[:, steps] = np.maximum(np.zeros(steps + 1), strike_price - price_tree[:, steps])
    for i in np.arange(steps - 1, -1, -1):
        for j in np.arange(0, i + 1):
            option_tree[j, i] = np.exp(-interest_rate * dt) * (p * option_tree[j, i + 1] + q * option_tree[j + 1, i + 1])
    return option_tree[0, 0]

def calculate_futures_price(spot_price, risk_free_rate, time_to_maturity):
    futures_price = spot_price * np.exp(risk_free_rate * time_to_maturity)
    return futures_price

def generate_price_paths(mid_price_func, num_paths, num_steps, drift, volatility, time):
    dt = time / num_steps
    price_paths = []
    for _ in range(num_paths):
        price_path = [mid_price_func]  # Initial mid-price
        for _ in range(num_steps):
            z = np.random.standard_normal()
            price = price_path[-1] * np.exp((drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * z)
            price_path.append(price)
        price_paths.append(price_path)
    return np.array(price_paths)

def risk_metrics_calc(risk_metrics, S=100, K=100, T=1, r=0.05, sigma=0.25):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if risk_metrics == 'Delta':
        delta = norm.cdf(d1)
        risk_metrics_calculation = f"The Delta Value: {delta}"
        return risk_metrics_calculation
    elif risk_metrics == 'Gamma':
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        risk_metrics_calculation = f"The Gamma Value: {gamma}"
        return risk_metrics_calculation
    elif risk_metrics == 'Vega':
        vega = S * norm.pdf(d1) * np.sqrt(T)
        risk_metrics_calculation = f"The Vega Value: {vega}"
        return risk_metrics_calculation
    elif risk_metrics == 'Theta':
        theta = - (S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))) - r * K * np.exp(-r * T) * norm.cdf(d2)
        risk_metrics_calculation = f"The Theta Value: {theta}"
        return risk_metrics_calculation

def main():
    st.title('Financial Derivative Calculator with Risk Analysis')
    st.markdown('---')
    st.sidebar.header('Select Derivative Type')
    derivative_type = st.sidebar.selectbox('Select Derivative:', ['Option', 'Futures', 'Swaps'])

    # Sidebar - Derivative Type and Input Parameters
    st.sidebar.header('Input Parameters')
    if derivative_type == 'Option':
        underlying_price = st.sidebar.number_input('Underlying Asset Price', value=100.0)
        strike_price = st.sidebar.number_input('Strike Price', value=110.0)
        volatility = st.sidebar.number_input('Volatility', value=0.2)
        interest_rate = st.sidebar.number_input('Interest Rate', value=0.05)
        time_to_expiry = st.sidebar.number_input('Time to Expiry (in years)', value=1.0)
        dividend_yield = st.sidebar.number_input('Dividend Yield', value=0.0)
        option_type = st.sidebar.selectbox('Option Type:', ['Call', 'Put'])
        option_pricing_model = st.sidebar.selectbox('Option Pricing Model:', ['Black-Scholes', 'Binomial Tree'])
        if option_pricing_model == 'Black-Scholes':
                priceBlack = calculate_black_scholes(underlying_price, strike_price, volatility, interest_rate, time_to_expiry, dividend_yield, option_type)
                st.write(f"The calculated price using Black-Scholes model is: {priceBlack}")
        elif option_pricing_model == 'Binomial Tree':
                priceBinomial = calculate_binomial_tree(underlying_price=100, strike_price=100, volatility=0.2, interest_rate=0.05, time_to_expiry=1, dividend_yield=0.03, option_type='Call', steps=100)
                st.write(f"The calculated price using Binomial Tree model is: {priceBinomial}")
        st.sidebar.header('Risk Metrics')
        risk_metrics = st.sidebar.selectbox('Select Risk Metrics:', ['Delta', 'Gamma', 'Vega', 'Theta'])
        risk_metrics_calculation = risk_metrics_calc(risk_metrics, underlying_price, strike_price, time_to_expiry, interest_rate, volatility)
        st.write(risk_metrics_calculation)

    elif derivative_type == 'Futures':
        futures_expiry_date = st.sidebar.date_input("Enter the futures expiry date (YYYY-MM-DD)")
        current_price = st.sidebar.number_input("Enter the current price of the underlying asset", value=101.0)
        risk_free_rate = st.sidebar.number_input("Enter the risk-free rate", value=0.05)
        current_date = date.today()
        time_to_maturity = (futures_expiry_date - current_date).days / 365.25
        future_price  = calculate_futures_price(current_price, risk_free_rate, time_to_maturity)
        st.write(f"The calculated futures price is: {future_price}")

    elif derivative_type == 'Swaps':
        swaps_notional_amount = st.sidebar.number_input("Enter the notional amount for the swap")
        swaps_fixed_rate = st.sidebar.number_input("Enter the fixed interest rate for the swap")
        swaps_maturity_date = st.sidebar.date_input("Enter the swap's maturity date (YYYY-MM-DD)")
        risk_free_rate = st.sidebar.number_input("Enter the risk-free rate", value=0.05)
        variable_rate_index = st.sidebar.number_input("Enter the current value of the variable rate index")
        current_date = date.today()
        time_to_maturity = (swaps_maturity_date - current_date).days / 365.25
        discount_factor = np.exp(-risk_free_rate * time_to_maturity)
        fixed_leg_pv = swaps_notional_amount * swaps_fixed_rate * discount_factor
        floating_leg_pv = swaps_notional_amount * variable_rate_index * discount_factor
        swap_npv = fixed_leg_pv - floating_leg_pv
        st.write(f"The calculated Net Present Value (NPV) of the swap is: {swap_npv}")

    st.sidebar.title('Monte Carlo Simulation')
    num_simulations = st.sidebar.number_input("Number of simulated price paths", value=10)
    num_time_steps = st.sidebar.number_input("Number of time steps", value=100)
    drift_value = st.sidebar.number_input("Drift parameter for GBM", value=0.1)
    volatility_value = st.sidebar.number_input("Volatility parameter for GBM", value=0.2)
    total_time = st.sidebar.number_input("Total time horizon for simulation", value=1.0)
    if derivative_type == 'Option':
        if option_pricing_model == 'Black-Scholes':
            simple_price = priceBlack
        else:
            simple_price = priceBinomial
    elif derivative_type == 'Futures':
        simple_price = future_price
    elif derivative_type == 'Swaps':
        simple_price = swap_npv
    price_paths = generate_price_paths(simple_price, num_simulations, num_time_steps, drift_value, volatility_value, total_time)
    plt.figure(figsize=(10, 5))
    for i in range(price_paths.shape[0]):
        plt.plot(price_paths[i], label=f'Path {i+1}')

    plt.title('Price Paths')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')

    st.markdown('---')
    st.header('Monte Carlo Simulation')
    st.pyplot(plt)

if __name__ == '__main__':
    main()
