import yfinance as yf
import pandas as pd
import numpy as np
from math import log, sqrt, exp
from scipy.stats import norm
from jax import grad
from jax.scipy.stats import norm as jax_norm
from datetime import datetime, time, date, timedelta
import matplotlib.pyplot as plt
import streamlit as st

#Pull data from Yahoo Finance
def fetch_yf_data(ticker:str, start:datetime, end:datetime) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, threads=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

# Black-Scholes Formula
def black_scholes(S:float, K:float, T:float, r:float, sigma:float, option_type) -> float:
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    else:
        price = K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return price

# Black-Scholes Formula using JAX for automatic differentiation
def black_scholes_jax(S:float, K:float, T:float, r:float, sigma:float, option_type) -> float:
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if option_type == "call":
        price = S * jax_norm.cdf(d1) - K * exp(-r * T) * jax_norm.cdf(d2)
    else:
        price = K * exp(-r * T) * jax_norm.cdf(-d2) - S * jax_norm.cdf(-d1)

    return price

# Loss Function and its Gradient
def loss(S:float, K:float, T, r:float, sigma_guess:float, price:float, option_type) -> float:
        guess_price = black_scholes_jax(S, K, T, r, sigma_guess, option_type)

        market_price = price

        return guess_price - market_price

# Gradient of the loss function with respect to sigma_guess
loss_gradient = grad(loss, argnums=4)

# Implied Volatility Solver using Newton-Raphson method
def IV_Solver(S:float, K:float, T:float, r:float, price:float, sigma_guess:float, option_type, iterations = 40, epsilon = 1e-6) -> float:
        for i in range(iterations):
            loss_value = loss(S, K, T, r, sigma_guess, price, option_type)
            if abs(loss_value) < epsilon:
                return sigma_guess
            else:
                loss_gradient_value = loss_gradient(S, K, T, r, sigma_guess, price, option_type)
                sigma_guess -= loss_value / loss_gradient_value

#Greeks
# Delta Function
def delta(S:float, K:float, T:float, r:float, sigma:float, option_type) -> float:
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    if option_type == "call":
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1
# Gamma Function
def gamma(S:float, K:float, T:float, r:float, sigma:float) -> float:
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    return norm.pdf(d1) / (S * sigma * sqrt(T))
# Vega Function
def vega(S:float, K:float, T:float, r:float, sigma:float) -> float:
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    return S * norm.pdf(d1) * sqrt(T)
# Theta Function
def theta(S:float, K:float, T:float, r:float, sigma:float, option_type) -> float:
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    
    if option_type == "call":
        theta_value = -(S * norm.pdf(d1) * sigma)/(2 * sqrt(T)) - (r * K * exp(-r * T) * norm.cdf(d2))
    else:
        theta_value = -(S * norm.pdf(d1) * sigma)/(2 * sqrt(T)) + (r * K * exp(-r * T) * norm.cdf(-d2))
    return theta_value
# Rho Function
def rho(S:float, K:float, T:float, r:float, sigma:float, option_type) -> float:
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if option_type == "call":
        rho_value = K * T * exp(-r * T) * norm.cdf(d2)
    else:
        rho_value = -K * T * exp(-r * T) * norm.cdf(-d2)
    return rho_value

# Heatmap Data Generators
def spot_sigma_heatmap_data(strike_price, time, riskfreerate, min_spot, max_spot, min_vol, max_vol, option_type):
    S_vals = np.linspace(min_spot, max_spot, 10)
    sigma_vals = np.linspace(min_vol, max_vol, 10)

    data = []
    for sigma in sigma_vals:
        row = [black_scholes(S, strike_price, time, riskfreerate, sigma, option_type) for S in S_vals]
        data.append(row)

    return pd.DataFrame(data, index=np.round(sigma_vals, 2), columns=np.round(S_vals, 2))

def spot_time_heatmap_data(strike_price, sigma, riskfreerate, min_spot, max_spot, min_time, max_time, option_type):
    S_vals = np.linspace(min_spot, max_spot, 10)
    time_vals = np.linspace(min_time, max_time, 10)

    data = []
    for T in time_vals:
        row = [black_scholes(S, strike_price, T, riskfreerate, sigma, option_type) for S in S_vals]
        data.append(row)

    return pd.DataFrame(data, index=np.round(time_vals, 2), columns=np.round(S_vals, 2))
    
def spot_sigma_greeks_heatmap_data(strike_price, time, riskfreerate, min_spot, max_spot, min_vol, max_vol, greek_type, option_type):
    S_vals = np.linspace(min_spot, max_spot, 10)
    sigma_vals = np.linspace(min_vol, max_vol, 10)

    data = []
    for sigma in sigma_vals:
        row = []
        for S in S_vals:
            if greek_type == "vega":
                value = vega(S, strike_price, time, riskfreerate, sigma) / 100
            elif greek_type == "gamma":
                value = gamma(S, strike_price, time, riskfreerate, sigma)
            elif greek_type == "delta":
                value = delta(S, strike_price, time, riskfreerate, sigma, option_type)
            else:
                raise ValueError("Unsupported Greek type")
            row.append(value)
        data.append(row)

    return pd.DataFrame(data, index=np.round(sigma_vals, 2), columns=np.round(S_vals, 2))

def spot_time_greeks_heatmap_data(strike_price, sigma, riskfreerate, min_spot, max_spot, min_time, max_time, greek_type, option_type):
    S_vals = np.linspace(min_spot, max_spot, 10)
    time_vals = np.linspace(min_time, max_time, 10)

    data = []
    for T in time_vals:
        row = []
        for S in S_vals:
            if greek_type == "theta":
                value = theta(S, strike_price, T, riskfreerate, sigma, option_type) / 365
            else:
                raise ValueError("Unsupported Greek type")
            row.append(value)
        data.append(row)

    return pd.DataFrame(data, index=np.round(time_vals, 2), columns=np.round(S_vals, 2))

# 3D Surface Data Generators
def generate_surface_data_spot_sigma(strike_price, riskfreerate, min_spot, max_spot, min_vol, max_vol, time, option_type, greek_bool, surface_type):
    S_vals = np.linspace(min_spot, max_spot, 30)
    sigma_vals = np.linspace(min_vol, max_vol, 30)
    x, y = np.meshgrid(S_vals, sigma_vals)
    
    if greek_bool:
        if surface_type == "Delta (S vs σ)":
            z = np.array([[delta(S, strike_price, time, riskfreerate, sigma, option_type) for S in S_vals] for sigma in sigma_vals])
        elif surface_type == "Gamma (S vs σ)":
            z = np.array([[gamma(S, strike_price, time, riskfreerate, sigma) for S in S_vals] for sigma in sigma_vals])
        elif surface_type == "Vega (S vs σ)":
            z = np.array([[vega(S, strike_price, time, riskfreerate, sigma) / 100 for S in S_vals] for sigma in sigma_vals])
        elif surface_type == "Rho (S vs σ)":
            z = np.array([[rho(S, strike_price, time, riskfreerate, sigma, option_type) / 100 for S in S_vals] for sigma in sigma_vals])
        else:
            raise ValueError("Unsupported Greek type")
    else:
        if surface_type == "Price (S vs σ)":
            z = np.array([[black_scholes(S, strike_price, time, riskfreerate, sigma, option_type) for S in S_vals] for sigma in sigma_vals])
        else:
            raise ValueError("Unsupported surface type")
    
    return x, y, z

def generate_surface_data_spot_time(strike_price, sigma, riskfreerate, min_spot, max_spot, min_time, max_time, option_type, greek_bool, surface_type):
    S_vals = np.linspace(min_spot, max_spot, 30)
    time_vals = np.linspace(min_time, max_time, 30)
    x, y = np.meshgrid(S_vals, time_vals)
    
    if greek_bool:
        if surface_type == "Delta (S vs T)":
            z = np.array([[delta(S, strike_price, T, riskfreerate, sigma, option_type) for S in S_vals] for T in time_vals])
        elif surface_type == "Gamma (S vs T)":
            z = np.array([[gamma(S, strike_price, T, riskfreerate, sigma) for S in S_vals] for T in time_vals])
        elif surface_type == "Vega (S vs T)":
            z = np.array([[vega(S, strike_price, T, riskfreerate, sigma) / 100 for S in S_vals] for T in time_vals])
        elif surface_type == "Theta (S vs T)":
            z = np.array([[theta(S, strike_price, T, riskfreerate, sigma, option_type) / 365 for S in S_vals] for T in time_vals])
        elif surface_type == "Rho (S vs T)":
            z = np.array([[rho(S, strike_price, T, riskfreerate, sigma, option_type) / 100 for S in S_vals] for T in time_vals])
        else:
            raise ValueError("Unsupported Greek type")
    else:
        if surface_type == "Price (S vs T)":
            z = np.array([[black_scholes(S, strike_price, T, riskfreerate, sigma, option_type) for S in S_vals] for T in time_vals])
        else:
            raise ValueError("Unsupported surface type")
    
    return x, y, z


# Payoff Calculation for Multi-Leg Strategies
def calculate_payoff(legs, S_range):
    payoff = np.zeros_like(S_range, dtype=float)

    for leg in legs:
        if leg["type"] == "stock":
            entry = leg.get("entry") 
            pnl = (S_range - entry) * leg["position"]
            payoff += pnl * leg["quantity"]
            continue

        if leg["type"] == "call":
            leg_payoff = np.maximum(S_range - leg["strike"], 0)
        else:  
            leg_payoff = np.maximum(leg["strike"] - S_range, 0)

        if leg["position"] == 1:  
            pnl = leg_payoff - leg["premium"]
        else:  
            pnl = leg["premium"] - leg_payoff
        
        payoff += pnl * leg["quantity"]

    return payoff
