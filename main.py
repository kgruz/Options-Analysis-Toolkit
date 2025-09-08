import streamlit as st
from math import log, sqrt, exp
from scipy.stats import norm
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
from arch import arch_model
import jax.numpy as jnp 
from jax import grad
from jax.scipy.stats import norm as jax_norm
import blackscholes as bs
from datetime import datetime, time, date, timedelta
from core import fetch_yf_data, black_scholes, loss, loss_gradient, IV_Solver, delta, gamma, vega, theta, rho, generate_surface_data_spot_sigma, generate_surface_data_spot_time, spot_time_heatmap_data, spot_sigma_heatmap_data, spot_sigma_greeks_heatmap_data, spot_time_greeks_heatmap_data, calculate_payoff

st.set_page_config(
    page_title="Options Multi-Purpose Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded")

real_ticker = st.sidebar.checkbox("Use Real Ticker Data (Currently Does not Work for Heatmaps, Surfaces, & Option Strategies)", value=False)

tab = st.sidebar.radio("Select View", ["Black-Scholes", "Inverse Black-Scholes Volatility", "Greeks", "Heatmaps", "3D Surfaces", "Option Strategies", "Volatility Information"])


if tab == "Black-Scholes":

    st.header("Black-Scholes Option Pricing Model")
    if real_ticker:
        st.write("Using real ticker data from Yahoo Finance")
        st.sidebar.header("Real Ticker Inputs")
        with st.sidebar:
            K = st.number_input("Strike Price (K)", value=100.0, format="%.2f")
            T = st.number_input("Time to Maturity (T in years)", value=1.0, format="%.4f")
            r = st.number_input("Risk-free Rate (r)", value=0.05, format="%.4f")
            sigma = st.number_input("Volatility (σ)", value=0.2, format="%.4f")
            ticker = st.text_input("Enter Ticker Symbol", value="AAPL")
            ticker_button = st.button("Fetch Data")
        calculate_button = st.button("Calculate Option Price")
        if ticker_button:
            try:
                end_date = (datetime.now() - timedelta(days=0)).strftime("%Y-%m-%d")
                start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")

                df = fetch_yf_data(ticker, start_date, end_date)
                
                st.write("Raw downloaded data:")
                st.dataframe(df)

                if df.empty or "Close" not in df.columns:
                    st.error("No data found or missing 'Close' column.")
                else:
                    latest_close = float(df["Close"].iloc[-1])
                    st.success(f"Latest closing price for {ticker}: ${latest_close:.2f}")
                    S = latest_close
                    st.session_state['stock_data'] = df
                    st.session_state["S"] = S
            except Exception as e:
                st.error(f"Error fetching data for ticker {ticker}: {e}")
                S = 100.0
                K = 100.0
                T = 1.0
                r = 0.05
                sigma = 0.2

                #stock_data = st.session_state['stock_data']
                #st.write("Downloaded Data:")
                #st.dataframe(stock_data.tail())

                
                #st.write(f"Last closing price: {S:.2f}")

        if calculate_button and st.session_state.get("S") is not None:
            
            stock_data = st.session_state['stock_data']
            st.write("Raw downloaded data:")
            st.dataframe(stock_data)
            S = st.session_state["S"]
            st.success(f"Latest closing price for {ticker}: ${S:.2f}")
            st.write("Option Price")
            price_call = black_scholes(S, K, T, r, sigma, "call")
            price_put = black_scholes(S, K, T, r, sigma, "put")
            st.success(f"The call option price is: ${price_call:.2f}")
            st.success(f"The put option price is: ${price_put:.2f}")
    else:
        with st.sidebar:
            S = st.number_input("Spot Price (S)", value=100.0, format="%.2f")
            K = st.number_input("Strike Price (K)", value=100.0, format="%.2f")
            T = st.number_input("Time to Maturity (T in years)", value=1.0, format="%.4f")
            r = st.number_input("Risk-free Rate (r)", value=0.05, format="%.4f")
            sigma = st.number_input("Volatility (σ)", value=0.2, format="%.4f")


        # Black-Scholes Function
        st.write("Option Price")
        price_call = black_scholes(S, K, T, r, sigma, "call")
        price_put = black_scholes(S, K, T, r, sigma, "put")
        st.success(f"The call option price is: ${price_call:.2f}")
        st.success(f"The put option price is: ${price_put:.2f}")

elif tab == "Inverse Black-Scholes Volatility":

    st.header("Inverse Black-Scholes Volatility Calculation")
    if real_ticker:
        st.write("Using real ticker data from Yahoo Finance")
        st.sidebar.header("Real Ticker Inputs")
        with st.sidebar:
            price = st.number_input("Option Price", value=10.0, format="%.2f")
            K = st.number_input("Strike Price (K)", value=100.0, format="%.2f")
            T = st.number_input("Time to Maturity (T in years)", value=1.0, format="%.4f")
            r = st.number_input("Risk-free Rate (r)", value=0.05, format="%.4f")
            sigma_guess = st.number_input("Initial Volatility Guess (σ)", value=0.2, format="%.4f")
            ticker = st.text_input("Enter Ticker Symbol", value="AAPL")
            ticker_button = st.button("Fetch Data")
        calculate_button = st.button("Calculate Implied Volatility")

        st.write("Inverse Black-Scholes Volatility")

        if ticker_button:
            try:
                # Try to pull 5 days of recent data
                end_date = (datetime.now() - timedelta(days=0)).strftime("%Y-%m-%d")
                start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")

                df = fetch_yf_data(ticker, start_date, end_date)
                
                st.write("Raw downloaded data:")
                st.dataframe(df)

                if df.empty or "Close" not in df.columns:
                    st.error("No data found or missing 'Close' column.")
                else:
                    latest_close = float(df["Close"].iloc[-1])
                    st.success(f"Latest closing price for {ticker}: ${latest_close:.2f}")
                    S = latest_close
                    st.session_state['stock_data'] = df
                    st.session_state["S"] = S
            except Exception as e:
                st.error(f"Error fetching data for ticker {ticker}: {e}")
                price = 10.0
                S = 100.0
                K = 100.0
                T = 1.0
                r = 0.05
                sigma_guess = 0.2
        if calculate_button and st.session_state.get("S") is not None:
            stock_data = st.session_state['stock_data']
            st.write("Raw downloaded data:")
            st.dataframe(stock_data)
            S = st.session_state["S"]
            st.success(f"Latest closing price for {ticker}: ${S:.2f}")
            sigma_inverse_call = IV_Solver(S, K, T, r, price, sigma_guess, "call")
            if sigma_inverse_call is not None:
                st.success(f"The implied volatility for the call option is: {sigma_inverse_call:.4f}")
            else:
                from scipy.optimize import brentq
                def objective_function(sigma):
                    return black_scholes(S, K, T, r, sigma, "call") - price
                try:
                    sigma_inverse = brentq(objective_function, 1e-6, 10)
                    st.success(f"The implied volatility for the call option is: {sigma_inverse:.4f}")
                except ValueError:
                    st.error("Could not find a valid implied volatility. Please check your inputs.")
            sigma_inverse_put = IV_Solver(S, K, T, r, price, sigma_guess, "put")
            if sigma_inverse_put is not None:
                st.success(f"The implied volatility for the put option is: {sigma_inverse_put:.4f}")
            else:
                from scipy.optimize import brentq
                def objective_function_put(sigma):
                    return black_scholes(S, K, T, r, sigma, "put") - price
                try:
                    sigma_inverse_put = brentq(objective_function_put, 1e-6, 10)
                    st.success(f"The implied volatility for the put option is: {sigma_inverse_put:.4f}")
                except ValueError:
                    st.error("Could not find a valid implied volatility for put option. Please check your inputs.")
    else:
        with st.sidebar:
            price = st.number_input("Option Price", value=10.0, format="%.2f")
            S = st.number_input("Spot Price (S)", value=100.0, format="%.2f")
            K = st.number_input("Strike Price (K)", value=100.0, format="%.2f")
            T = st.number_input("Time to Maturity (T in years)", value=1.0, format="%.4f")
            sigma_guess = st.number_input("Initial Volatility Guess (σ)", value=0.2, format="%.4f")
            r = st.number_input("Risk-free Rate (r)", value=0.05, format="%.4f")
            

        st.write("Inverse Black-Scholes Volatility")
        calculate = st.button("Calculate Implied Volatility")
        if calculate:
            sigma_inverse_call = IV_Solver(S, K, T, r, price, sigma_guess, "call")
            if sigma_inverse_call is not None:
                st.success(f"The implied volatility for the call option is: {sigma_inverse_call:.4f}")
            else:
                from scipy.optimize import brentq
                def objective_function(sigma):
                    return black_scholes(S, K, T, r, sigma, "call") - price
                try:
                    sigma_inverse = brentq(objective_function, 1e-6, 10)
                    st.success(f"The implied volatility for the call option is: {sigma_inverse:.4f}")
                except ValueError:
                    st.error("Could not find a valid implied volatility. Please check your inputs.")
            sigma_inverse_put = IV_Solver(S, K, T, r, price, sigma_guess, "put")
            if sigma_inverse_put is not None:
                st.success(f"The implied volatility for the put option is: {sigma_inverse_put:.4f}")
            else:
                from scipy.optimize import brentq
                def objective_function_put(sigma):
                    return black_scholes(S, K, T, r, sigma, "put") - price
                try:
                    sigma_inverse_put = brentq(objective_function_put, 1e-6, 10)
                    st.success(f"The implied volatility for the put option is: {sigma_inverse_put:.4f}")
                except ValueError:
                    st.error("Could not find a valid implied volatility for put option. Please check your inputs.")

elif tab == "Greeks":
    
    def plot_greeks(option_type_plot):
        S_vals = np.linspace(K * 0.5, K * 1.5, 100)
        deltas = [delta(S_i, K, T, r, sigma, option_type_plot) for S_i in S_vals]
        gammas = [gamma(S_i, K, T, r, sigma) for S_i in S_vals]
        vegas = [vega(S_i, K, T, r, sigma) / 100 for S_i in S_vals]  # per 1%
        thetas = [theta(S_i, K, T, r, sigma, option_type_plot) / 365 for S_i in S_vals]
        rhos   = [rho(S_i, K, T, r, sigma, option_type_plot) / 100 for S_i in S_vals]  # per 1%

        fig, ax = plt.subplots()
        ax.plot(S_vals, deltas, label='Delta')
        ax.plot(S_vals, gammas, label='Gamma')
        ax.plot(S_vals, vegas, label='Vega')
        ax.plot(S_vals, thetas, label='Theta')
        ax.plot(S_vals, rhos, label='Rho')

        ax.set_title("Greeks vs Spot Price")
        ax.set_xlabel("Spot Price (S)")
        ax.set_ylabel("Greek Value")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    st.header("Option Greeks")
    if real_ticker:
        st.write("Using real ticker data from Yahoo Finance")
        st.sidebar.header("Real Ticker Inputs")
        with st.sidebar:
            K = st.number_input("Strike Price (K)", value=100.0, format="%.2f")
            T = st.number_input("Time to Maturity (T in years)", value=1.0, format="%.4f")
            r = st.number_input("Risk-free Rate (r)", value=0.05, format="%.4f")
            sigma = st.number_input("Volatility (σ)", value=0.2, format="%.4f")
            ticker = st.text_input("Enter Ticker Symbol", value="AAPL")
            ticker_button = st.button("Fetch Data")
        calculate_button = st.button("Calculate Greeks Information")

        st.write("Option Greeks Information")

        if ticker_button:
            try:
                # Try to pull 5 days of recent data
                end_date = (datetime.now() - timedelta(days=0)).strftime("%Y-%m-%d")
                start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")

                df = fetch_yf_data(ticker, start_date, end_date)
                
                st.write("Raw downloaded data:")
                st.dataframe(df)

                if df.empty or "Close" not in df.columns:
                    st.error("No data found or missing 'Close' column.")
                else:
                    latest_close = float(df["Close"].iloc[-1])
                    st.success(f"Latest closing price for {ticker}: ${latest_close:.2f}")
                    S = latest_close
                    st.session_state['stock_data'] = df
                    st.session_state["S"] = S
            except Exception as e:
                st.error(f"Error fetching data for ticker {ticker}: {e}")
                S = 100.0
                K = 100.0
                T = 1.0
                r = 0.05
                sigma = 0.2
        if calculate_button and st.session_state.get("S") is not None:
            stock_data = st.session_state['stock_data']
            st.write("Raw downloaded data:")
            st.dataframe(stock_data)
            S = st.session_state["S"]
            st.success(f"Latest closing price for {ticker}: ${S:.2f}")
            # Calculate Greeks
            delta_call = delta(S, K, T, r, sigma, "call")
            delta_put = delta(S, K, T, r, sigma, "put")
            gamma_val = gamma(S, K, T, r, sigma)
            vega_val = vega(S, K, T, r, sigma) / 100
            theta_call = theta(S, K, T, r, sigma, "call")
            theta_put = theta(S, K, T, r, sigma, "put")
            rho_call = rho(S, K, T, r, sigma, "call")
            rho_put = rho(S, K, T, r, sigma, "put")
            st.write("Deltas")
            st.success(f"The delta of the call option is: {delta_call:.4f}")
            st.success(f"The delta of the put option is: {delta_put:.4f}")
            st.write("Gamma")
            st.success(f"The gamma of the option is: {gamma_val:.4f}")
            st.write("Vega")
            st.success(f"The vega per 1% change in volatility of underlying for the option is: {vega_val:.4f}")
            st.write("Thetas")
            st.success(f"The theta per day of the call option is: {theta_call/365:.4f}")
            st.success(f"The theta per day of the put option is: {theta_put/365:.4f}")
            st.write("Rhos")
            st.success(f"The rho per 1% change in interest rate for the call option is: {rho_call/100:.4f}")
            st.success(f"The rho per 1% change in interest rate for the put option is: {rho_put/100:.4f}")
            # Plot Greeks
            st.markdown("---")
            st.write("Greeks Plots")
            col1, col2 = st.columns([1, 1], gap = "small")
            with col1:
                st.markdown("Greeks Plotted for Call Option")
                plot_greeks("call")
            with col2:
                st.markdown("Greeks Plotted for Put Option")
                plot_greeks("put")


    else:
        with st.sidebar:
            S = st.number_input("Spot Price (S)", value=100.0, format="%.2f")
            K = st.number_input("Strike Price (K)", value=100.0, format="%.2f")
            T = st.number_input("Time to Maturity (T in years)", value=1.0, format="%.4f")
            r = st.number_input("Risk-free Rate (r)", value=0.05, format="%.4f")
            sigma = st.number_input("Volatility (σ)", value=0.2, format="%.4f")

    # Output

        #if st.button("Obtain Delta Information"):
        st.write("Deltas")
        delta_call = delta(S, K, T, r, sigma, "call")
        delta_put = delta(S, K, T, r, sigma, "put")
        st.success(f"The delta of the call option is: {delta_call:.4f}")
        st.success(f"The delta of the put option is: {delta_put:.4f}")

        #if st.button("Obtain Gamma Information"):
        st.write("Gamma")
        gamma_val = gamma(S, K, T, r, sigma)
        st.success(f"The gamma of the option is: {gamma_val:.4f}")

        #if st.button("Obtain Vega Information"):
        st.write("Vega")
        vega_val = vega(S, K, T, r, sigma)
        st.success(f"The vega per 1% change in volatility of underlying for the option is: {vega_val/100:.4f}")

        #if st.button("Obtain Theta Information"):
        st.write("Thetas")
        theta_call = theta(S, K, T, r, sigma, "call")
        theta_put = theta(S, K, T, r, sigma, "put")
        st.success(f"The theta per day of the call option is: {theta_call/365:.4f}")
        st.success(f"The theta per day of the call option is: {theta_put/365:.4f}")

        #if st.button("Obtain Rho Information"):
        st.write("Rhos")
        rho_call = rho(S, K, T, r, sigma, "call")
        rho_put = rho(S, K, T, r, sigma, "put")
        st.success(f"The rho per 1% change in interest rate for the call option is: {rho_call/100:.4f}")
        st.success(f"The rho per 1% change in interest rate for the put option is: {rho_put/100:.4f}")

        #if st.button("Plot Greeks Info"):
        st.markdown("---")
        st.write("Greeks Plots")



        col1, col2 = st.columns([1, 1], gap = "small")
        with col1:
            st.markdown("Greeks Plotted for Call Option")
            plot_greeks("call")


        with col2:
            st.markdown("Greeks Plotted for Put Option")
            plot_greeks("put")

elif tab == "Heatmaps":
    st.header("Heatmaps")
    
    heatmap_type = st.sidebar.selectbox("Select Heatmap Type", ["Price vs Spot & Volatility", "Price vs Spot & Time", "Vega vs Spot & Volatility", "Gamma vs Spot & Volatility", "Theta vs Spot & Time", "Delta vs Spot and Volatility"])

    if heatmap_type == "Price vs Spot & Volatility" or heatmap_type == "Vega vs Spot & Volatility" or heatmap_type == "Gamma vs Spot & Volatility" or heatmap_type == "Delta vs Spot and Volatility":
        with st.sidebar:
            strike_price = st.number_input("Strike Price", value = 100.0, format="%.2f")
            time = st.number_input("Time to Maturity", value = 1.0, format="%.4f")
            riskfreerate = st.number_input("Risk-Free Interest Rate", value = 0.05, format="%.4f")
            min_spot = st.number_input("Minimum Spot Price", value = 80.0, format="%.2f")
            max_spot = st.number_input("Maximum Spot Price", value = 120.0, format="%.2f")
            min_vol = st.slider("Minimum Volatility", min_value = 0.00, max_value = 1.00, value = 0.10, format="%.4f")
            max_vol = st.slider("Maximum Volatility", min_value = 0.00, max_value = 1.00, value = 0.50, format="%.4f")
    elif heatmap_type == "Price vs Spot & Time" or heatmap_type == "Theta vs Spot & Time":
        with st.sidebar:
            strike_price = st.number_input("Strike Price", value = 100.0, format="%.2f")
            sigma = st.number_input("Volatility", value = 0.2, format="%.4f")
            riskfreerate = st.number_input("Risk-Free Interest Rate", value = 0.05, format="%.4f")
            min_spot = st.number_input("Minimum Spot Price", value = 80.0, format="%.2f")
            max_spot = st.number_input("Maximum Spot Price", value = 120.0, format="%.2f")
            min_time = st.number_input("Minimum Time to Maturity (in years)", value = 0.01, format="%.4f")
            max_time = st.number_input("Maximum Time to Maturity (in years)", value = 5.0, format="%.4f")
    

    if heatmap_type == "Vega vs Spot & Volatility":
        st.markdown("Vega Heatmap for Spot Price vs. Volatility")
        df_vega = spot_sigma_greeks_heatmap_data(strike_price, time, riskfreerate, min_spot, max_spot, min_vol, max_vol, "vega", "call")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df_vega, annot = True, fmt=".2f", cmap = "coolwarm", linewidths = 0.5, ax = ax)
        ax.set_xlabel("Spot Price")
        ax.set_ylabel("Volatility")
        ax.set_title(f"Vega Heatmap (S vs σ) at given parameters")
        st.pyplot(fig)
    elif heatmap_type == "Gamma vs Spot & Volatility":
        st.markdown("Gamma Heatmap for Spot Price vs. Volatility")
        df_gamma = spot_sigma_greeks_heatmap_data(strike_price, time, riskfreerate, min_spot, max_spot, min_vol, max_vol, "gamma", "call")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df_gamma, annot = True, fmt=".2f", cmap = "coolwarm", linewidths = 0.5, ax = ax)
        ax.set_xlabel("Spot Price")
        ax.set_ylabel("Volatility")
        ax.set_title(f"Gamma Heatmap (S vs σ) at given parameters")
        st.pyplot(fig)
    else:
        col1, col2 = st.columns([1, 1], gap = "small")
        with col1:
            if heatmap_type == "Price vs Spot & Volatility":
                st.markdown("Call Option Heatmap for Spot Price vs. Volatility")
                df_sigma_vol_call = spot_sigma_heatmap_data(strike_price, time, riskfreerate, min_spot, max_spot, min_vol, max_vol, "call")
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(df_sigma_vol_call, annot = True, fmt=".2f", cmap = "coolwarm", linewidths = 0.5, ax = ax)
                ax.set_xlabel("Spot Price")
                ax.set_ylabel("Volatility")
                ax.set_title(f"Call Price Heatmap (S vs σ) at given parameters")
                st.pyplot(fig)
            elif heatmap_type == "Price vs Spot & Time":
                st.markdown("Call Option Heatmap for Spot Price vs. Time")
                df_spot_time_call = spot_time_heatmap_data(strike_price, sigma, riskfreerate, min_spot, max_spot, min_time, max_time, "call")
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(df_spot_time_call, annot = True, fmt=".2f", cmap = "coolwarm", linewidths = 0.5, ax = ax)
                ax.set_xlabel("Spot Price")
                ax.set_ylabel("Time to Maturity")
                ax.set_title(f"Call Price Heatmap (S vs T) at given parameters")
                st.pyplot(fig)
            elif heatmap_type == "Theta vs Spot & Time":
                st.markdown("Theta Heatmap for Spot Price vs. Time")
                df_theta = spot_time_greeks_heatmap_data(strike_price, sigma, riskfreerate, min_spot, max_spot, min_time, max_time, "theta", "call")
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(df_theta, annot = True, fmt=".2f", cmap = "coolwarm", linewidths = 0.5, ax = ax)
                ax.set_xlabel("Spot Price")
                ax.set_ylabel("Time to Maturity")
                ax.set_title(f"Theta Heatmap (S vs T) at given parameters")
                st.pyplot(fig)
            elif heatmap_type == "Delta vs Spot and Volatility":
                st.markdown("Delta Heatmap for Spot Price vs. Volatility")
                df_delta = spot_sigma_greeks_heatmap_data(strike_price, time, riskfreerate, min_spot, max_spot, min_vol, max_vol, "delta", "call")
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(df_delta, annot = True, fmt=".2f", cmap = "coolwarm", linewidths = 0.5, ax = ax)
                ax.set_xlabel("Spot Price")
                ax.set_ylabel("Volatility")
                ax.set_title(f"Delta Heatmap (S vs σ) at given parameters")
                st.pyplot(fig)
        with col2:
            if heatmap_type == "Price vs Spot & Volatility":
                st.markdown("Put Option Heatmap for Spot Price vs. Volatility")
                df_sigma_vol_put = spot_sigma_heatmap_data(strike_price, time, riskfreerate, min_spot, max_spot, min_vol, max_vol, "put")
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(df_sigma_vol_put, annot = True, fmt=".2f", cmap = "coolwarm", linewidths = 0.5, ax = ax)
                ax.set_xlabel("Spot Price")
                ax.set_ylabel("Volatility")
                ax.set_title(f"Put Price Heatmap (S vs σ) at given parameters")
                st.pyplot(fig)
            elif heatmap_type == "Price vs Spot & Time":
                st.markdown("Put Option Heatmap for Spot Price vs. Time")
                df_spot_time_put = spot_time_heatmap_data(strike_price, sigma, riskfreerate, min_spot, max_spot, min_time, max_time, "put")
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(df_spot_time_put, annot = True, fmt=".2f", cmap = "coolwarm", linewidths = 0.5, ax = ax)
                ax.set_xlabel("Spot Price")
                ax.set_ylabel("Time to Maturity")
                ax.set_title(f"Put Price Heatmap (S vs T) at given parameters")
                st.pyplot(fig)
            elif heatmap_type == "Theta vs Spot & Time":
                st.markdown("Theta Heatmap for Spot Price vs. Time")
                df_theta = spot_time_greeks_heatmap_data(strike_price, sigma, riskfreerate, min_spot, max_spot, min_time, max_time, "theta", "put")
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(df_theta, annot = True, fmt=".2f", cmap = "coolwarm", linewidths = 0.5, ax = ax)
                ax.set_xlabel("Spot Price")
                ax.set_ylabel("Time to Maturity")
                ax.set_title(f"Theta Heatmap (S vs T) at given parameters")
                st.pyplot(fig)
            elif heatmap_type == "Delta vs Spot and Volatility":
                st.markdown("Delta Heatmap for Spot Price vs. Volatility")
                df_delta = spot_sigma_greeks_heatmap_data(strike_price, time, riskfreerate, min_spot, max_spot, min_vol, max_vol, "delta", "put")
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(df_delta, annot = True, fmt=".2f", cmap = "coolwarm", linewidths = 0.5, ax = ax)
                ax.set_xlabel("Spot Price")
                ax.set_ylabel("Volatility")
                ax.set_title(f"Delta Heatmap (S vs σ) at given parameters")
                st.pyplot(fig)

elif tab == "3D Surfaces":
    st.header("3D Surfaces")
    
    surface_type = st.selectbox("Choose a 3D surface:", [
    "Price (S vs σ)", 
    "Price (S vs T)",
    "Delta (S vs σ)",
    "Delta (S vs T)", 
    "Gamma (S vs σ)",
    "Gamma (S vs T)", 
    "Theta (S vs T)",
    "Vega (S vs σ)",
    "Vega (S vs T)",
    "Rho (S vs σ)",
    "Rho (S vs T)"
    ])
    def plot_surface(x, y, z, title, x_label, y_label, z_label="Value"):
        fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale='thermal')])
        fig.update_layout(
            title=title,
            scene_camera=dict(
                eye=dict(x=-1.5, y=-1.5, z=1.5)
            ),
            autosize=False,
            width=600,
            height=600,
            margin=dict(l=65, r=50, b=65, t=90),
            scene=dict(
                xaxis_title=x_label,
                yaxis_title=y_label,
                zaxis_title=z_label
            )
        )
        st.plotly_chart(fig, use_container_width=False)

    if surface_type == "Price (S vs σ)" or surface_type == "Gamma (S vs σ)" or surface_type == "Vega (S vs σ)" or surface_type == "Delta (S vs σ)" or surface_type == "Rho (S vs σ)":
        with st.sidebar:
            strike_price = st.number_input("Strike Price", value = 100.0, format="%.2f")
            time = st.number_input("Time to Maturity", value = 1.0, format="%.4f")
            riskfreerate = st.number_input("Risk-Free Interest Rate", value = 0.05, format="%.4f")
            min_spot = st.number_input("Minimum Spot Price", value = 80.0, format="%.2f")
            max_spot = st.number_input("Maximum Spot Price", value = 120.0, format="%.2f")
            min_vol = st.slider("Minimum Volatility", min_value = 0.00, max_value = 1.00, value = 0.10, format="%.4f")
            max_vol = st.slider("Maximum Volatility", min_value = 0.00, max_value = 1.00, value = 0.50, format="%.4f")
    elif surface_type == "Price (S vs T)" or surface_type == "Gamma (S vs T)" or surface_type == "Theta (S vs T)" or surface_type == "Vega (S vs T)" or surface_type == "Delta (S vs T)" or surface_type == "Rho (S vs T)":
        with st.sidebar:
            strike_price = st.number_input("Strike Price", value = 100.0, format="%.2f")
            sigma = st.number_input("Volatility", value = 0.2, format="%.4f")
            riskfreerate = st.number_input("Risk-Free Interest Rate", value = 0.05, format="%.4f")
            min_spot = st.number_input("Minimum Spot Price", value = 80.0, format="%.2f")
            max_spot = st.number_input("Maximum Spot Price", value = 120.0, format="%.2f")
            min_time = st.number_input("Minimum Time to Maturity (in years)", value = 0.01, format="%.4f")
            max_time = st.number_input("Maximum Time to Maturity (in years)", value = 5.0, format="%.4f")
    
    if surface_type == "Gamma (S vs σ)":
        x, y, z = generate_surface_data_spot_sigma(strike_price, riskfreerate, min_spot, max_spot, min_vol, max_vol, time, "call", True, surface_type)
        plot_surface(x, y, z, title="Call Gamma Surface", x_label="S", y_label="σ")
    elif surface_type == "Gamma (S vs T)":
        x, y, z = generate_surface_data_spot_time(strike_price, sigma, riskfreerate, min_spot, max_spot, min_time, max_time, "call", True, surface_type)
        plot_surface(x, y, z, title="Call Gamma Surface", x_label="S", y_label="T")
    elif surface_type == "Vega (S vs σ)":
        x, y, z = generate_surface_data_spot_sigma(strike_price, riskfreerate, min_spot, max_spot, min_vol, max_vol, time, "call", True, surface_type)
        plot_surface(x, y, z, title="Call Vega Surface", x_label="S", y_label="σ")
    elif surface_type == "Vega (S vs T)":
        x, y, z = generate_surface_data_spot_time(strike_price, sigma, riskfreerate, min_spot, max_spot, min_time, max_time, "call", True, surface_type)
        plot_surface(x, y, z, title="Call Vega Surface", x_label="S", y_label="T")
    else:
        col1, col2 = st.columns([1, 1], gap = "small")
        with col1:
            if surface_type == "Price (S vs σ)":
                x, y, z = generate_surface_data_spot_sigma(strike_price, riskfreerate, min_spot, max_spot, min_vol, max_vol, time, "call", False, surface_type)
                plot_surface(x, y, z, title="Call Price Surface", x_label="S", y_label="σ")
            elif surface_type == "Price (S vs T)":
                x, y, z = generate_surface_data_spot_time(strike_price, sigma, riskfreerate, min_spot, max_spot, min_time, max_time, "call", False, surface_type)
                plot_surface(x, y, z, title="Call Price Surface", x_label="S", y_label="T")
            elif surface_type == "Delta (S vs σ)":
                x, y, z = generate_surface_data_spot_sigma(strike_price, riskfreerate, min_spot, max_spot, min_vol, max_vol, time, "call", True, surface_type)
                plot_surface(x, y, z, title="Call Delta Surface", x_label="S", y_label="σ")
            elif surface_type == "Delta (S vs T)":
                x, y, z = generate_surface_data_spot_time(strike_price, sigma, riskfreerate, min_spot, max_spot, min_time, max_time, "call", True, surface_type)
                plot_surface(x, y, z, title="Call Delta Surface", x_label="S", y_label="T")
            elif surface_type == "Theta (S vs T)":
                x, y, z = generate_surface_data_spot_time(strike_price, sigma, riskfreerate, min_spot, max_spot, min_time, max_time, "call", True, surface_type)
                plot_surface(x, y, z, title="Call Theta Surface", x_label="S", y_label="T")
            elif surface_type == "Rho (S vs σ)":
                x, y, z = generate_surface_data_spot_sigma(strike_price, riskfreerate, min_spot, max_spot, min_vol, max_vol, time, "call", True, surface_type)
                plot_surface(x, y, z, title="Call Rho Surface", x_label="S", y_label="σ")
            elif surface_type == "Rho (S vs T)":
                x, y, z = generate_surface_data_spot_time(strike_price, sigma, riskfreerate, min_spot, max_spot, min_time, max_time, "call", True, surface_type)
                plot_surface(x, y, z, title="Call Rho Surface", x_label="S", y_label="T")
        with col2:
            if surface_type == "Price (S vs σ)":
                x, y, z = generate_surface_data_spot_sigma(strike_price, riskfreerate, min_spot, max_spot, min_vol, max_vol, time, "put", False, surface_type)
                plot_surface(x, y, z, title="Put Price Surface", x_label="S", y_label="σ")
            elif surface_type == "Price (S vs T)":
                x, y, z = generate_surface_data_spot_time(strike_price, sigma, riskfreerate, min_spot, max_spot, min_time, max_time, "put", False, surface_type)
                plot_surface(x, y, z, title="Put Price Surface", x_label="S", y_label="T")
            elif surface_type == "Delta (S vs σ)":
                x, y, z = generate_surface_data_spot_sigma(strike_price, riskfreerate, min_spot, max_spot, min_vol, max_vol, time, "put", True, surface_type)
                plot_surface(x, y, z, title="Put Delta Surface", x_label="S", y_label="σ")
            elif surface_type == "Delta (S vs T)":
                x, y, z = generate_surface_data_spot_time(strike_price, sigma, riskfreerate, min_spot, max_spot, min_time, max_time, "put", True, surface_type)
                plot_surface(x, y, z, title="Put Delta Surface", x_label="S", y_label="T")
            elif surface_type == "Theta (S vs T)":
                x, y, z = generate_surface_data_spot_time(strike_price, sigma, riskfreerate, min_spot, max_spot, min_time, max_time, "put", True, surface_type)
                plot_surface(x, y, z, title="Put Theta Surface", x_label="S", y_label="T")
            elif surface_type == "Rho (S vs σ)":
                x, y, z = generate_surface_data_spot_sigma(strike_price, riskfreerate, min_spot, max_spot, min_vol, max_vol, time, "put", True, surface_type)
                plot_surface(x, y, z, title="Put Rho Surface", x_label="S", y_label="σ")
            elif surface_type == "Rho (S vs T)":
                x, y, z = generate_surface_data_spot_time(strike_price, sigma, riskfreerate, min_spot, max_spot, min_time, max_time, "put", True, surface_type)
                plot_surface(x, y, z, title="Put Rho Surface", x_label="S", y_label="T")

elif tab == "Option Strategies":
    st.header("Option Strategies")
    st.markdown("This section allows you to explore various option strategies using the Black-Scholes model.")
    st.markdown("Make sure your first strike price is lower than the second strike price for spreads.")
    
    def plot_strategy_payoff(legs, S_range, spot_price):
        payoff = calculate_payoff(legs, S_range)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=S_range[payoff >= 0],
            y=payoff[payoff >= 0],
            fill='tozeroy',
            fillcolor='rgba(0,200,0,0.15)',
            mode='none',
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=S_range[payoff <= 0],
            y=payoff[payoff <= 0],
            fill='tozeroy',
            fillcolor='rgba(200,0,0,0.15)',
            mode='none',
            showlegend=False
        ))

        # Main payoff line
        fig.add_trace(go.Scatter(
            x=S_range, 
            y=payoff, 
            mode='lines',
            line=dict(width=3, color='blue'),
            name='Net Payoff'
        ))

        for leg in legs:

            if leg["type"] == "stock":
                continue 

            if leg["type"] == "call":
                payoff_leg = max(leg["strike"] - leg["strike"], 0) if leg["position"] < 0 else max(0, leg["strike"] - leg["strike"])
            else:
                payoff_leg = max(leg["strike"] - leg["strike"], 0) if leg["position"] > 0 else -max(leg["strike"] - leg["strike"], 0)

            payoff_leg = leg["position"] * leg["quantity"] * (max(leg["strike"] - leg["strike"], 0) if leg["type"] == "put" 
                                                            else max(leg["strike"] - leg["strike"], 0)) - leg["premium"] * leg["position"] * leg["quantity"]
            
            payoff_total = calculate_payoff(legs, np.array([leg["strike"]]))[0]
            
            color = 'green' if leg["position"] > 0 else 'red'
            fig.add_trace(go.Scatter(
                x=[leg["strike"]],
                y=[payoff_total],
                mode='markers',
                marker=dict(color=color, size=10),
                name=("Long" if leg["position"] > 0 else "Short") + f" {leg['type'].capitalize()} @ {leg['strike']}"
            ))

        fig.add_shape(type='line', x0=S_range[0], x1=S_range[-1], y0=0, y1=0,
                    line=dict(color='black', dash='dash'))
        fig.add_shape(type='line', x0=spot_price, x1=spot_price,
                    y0=min(payoff), y1=max(payoff),
                    line=dict(color='gray', dash='dot'))
        
        fig.update_layout(
            title="Interactive Options Strategy Payoff",
            xaxis_title="Underlying Price (S)",
            yaxis_title="Profit / Loss",
            hovermode="x unified",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

    
    strategy_type = st.sidebar.selectbox("Select Strategy Type", ["Single Option", "Spread", "Straddle", "Strangle", "Butterfly", "Iron Condor"])
    
    legs = []
    spot_price = st.sidebar.number_input("Spot Price", value=100.0, format="%.2f")
    if not strategy_type == "Calendar Spread":
        T = st.sidebar.number_input("Time to Maturity (yrs)", value=1.0, format="%.4f")
    else:
        strike = st.sidebar.number_input("Strike Price", value=100.0, format="%.2f")
    r = st.sidebar.number_input("Risk-Free Rate", value=0.05, format="%.4f")
    sigma = st.sidebar.number_input("Volatility (σ)", value=0.2, format="%.4f")

    if strategy_type == "Single Option":
        option_choice = st.selectbox(
            "Select Option Type",
            ["Long Call", "Long Put", "Short Call", "Short Put", "Covered Call", "Cash-Secured Put"]
        )

        quantity = st.sidebar.number_input("Contracts / Shares (unit-matched)", value=1, step=1)

        if option_choice == "Covered Call":
            strike = st.sidebar.number_input("Call Strike Price", value=100.0)
            
            legs.append({"type": "stock", "position": 1, "entry": spot_price, "quantity": quantity})
            
            premium = black_scholes(spot_price, strike, T, r, sigma, "call")
            legs.append({"type": "call", "position": -1, "strike": strike, "premium": premium, "quantity": quantity})

        elif option_choice == "Cash-Secured Put":
            strike = st.sidebar.number_input("Put Strike Price", value=100.0)
            
            premium = black_scholes(spot_price, strike, T, r, sigma, "put")
            legs.append({"type": "put", "position": -1, "strike": strike, "premium": premium, "quantity": quantity})

        else:
            
            strike = st.sidebar.number_input("Strike Price", value=100.0)
            opt_type = "call" if "Call" in option_choice else "put"
            pos = 1 if "Long" in option_choice else -1
            premium = black_scholes(spot_price, strike, T, r, sigma, opt_type)
            legs.append({"type": opt_type, "position": pos, "strike": strike, "premium": premium, "quantity": quantity})
    if strategy_type == "Spread":
        spread_type = st.selectbox("Select Spread Type", 
                                        ["Bull Call Spread", "Bull Put Spread", 
                                            "Bear Call Spread", "Bear Put Spread"])
        
        strike_price1 = st.sidebar.number_input("Lower Strike Price", value=90.0, format="%.2f")
        strike_price2 = st.sidebar.number_input("Higher Strike Price", value=110.0, format="%.2f")
        quantity = st.sidebar.number_input("Contracts", value=1, step=1)

    
        if spread_type == "Bull Call Spread":
            # Buy lower strike call, sell higher strike call
            price1 = black_scholes(spot_price, strike_price1, T, r, sigma, "call")
            price2 = black_scholes(spot_price, strike_price2, T, r, sigma, "call")
            legs.append({"type": "call", "position": 1, "strike": strike_price1, "premium": price1, "quantity": quantity})
            legs.append({"type": "call", "position": -1, "strike": strike_price2, "premium": price2, "quantity": quantity})

        
        elif spread_type == "Bull Put Spread":
            # Sell higher strike put, buy lower strike put
            price1 = black_scholes(spot_price, strike_price2, T, r, sigma, "put")  # sell high strike
            price2 = black_scholes(spot_price, strike_price1, T, r, sigma, "put")  # buy low strike
            legs.append({"type": "put", "position": -1, "strike": strike_price2, "premium": price1, "quantity": quantity})
            legs.append({"type": "put", "position": 1, "strike": strike_price1, "premium": price2, "quantity": quantity})

        # --- BEAR CALL SPREAD (Credit Spread) ---
        elif spread_type == "Bear Call Spread":
            # Sell lower strike call, buy higher strike call
            price1 = black_scholes(spot_price, strike_price1, T, r, sigma, "call")  # sell low strike
            price2 = black_scholes(spot_price, strike_price2, T, r, sigma, "call")  # buy high strike
            legs.append({"type": "call", "position": -1, "strike": strike_price1, "premium": price1, "quantity": quantity})
            legs.append({"type": "call", "position": 1, "strike": strike_price2, "premium": price2, "quantity": quantity})

        
        elif spread_type == "Bear Put Spread":
            # Buy higher strike put, sell lower strike put
            price1 = black_scholes(spot_price, strike_price2, T, r, sigma, "put")  # buy high strike
            price2 = black_scholes(spot_price, strike_price1, T, r, sigma, "put")  # sell low strike
            legs.append({"type": "put", "position": 1, "strike": strike_price2, "premium": price1, "quantity": quantity})
            legs.append({"type": "put", "position": -1, "strike": strike_price1, "premium": price2, "quantity": quantity})
    elif strategy_type == "Straddle":
        straddle_type = st.selectbox("Select Straddle Type", ["Long Straddle", "Short Straddle"])

        strike = st.sidebar.number_input("Strike Price", value=100.0, format="%.2f")
        quantity = st.sidebar.number_input("Contracts", value=1, step=1)

        if straddle_type == "Long Straddle":
            price_call = black_scholes(spot_price, strike, T, r, sigma, "call")
            price_put = black_scholes(spot_price, strike, T, r, sigma, "put")
            legs.append({"type": "call", "position": 1, "strike": strike, "premium": price_call, "quantity": quantity})
            legs.append({"type": "put", "position": 1, "strike": strike, "premium": price_put, "quantity": quantity})
        elif straddle_type == "Short Straddle":
            price_call = black_scholes(spot_price, strike, T, r, sigma, "call")
            price_put = black_scholes(spot_price, strike, T, r, sigma, "put")
            legs.append({"type": "call", "position": -1, "strike": strike, "premium": price_call, "quantity": quantity})
            legs.append({"type": "put", "position": -1, "strike": strike, "premium": price_put, "quantity": quantity})
    elif strategy_type == "Strangle":
        strangle_type = st.selectbox("Select Strangle Type", ["Long Strangle", "Short Strangle"])

        strike_call = st.sidebar.number_input("Call Strike Price", value=110.0, format="%.2f")
        strike_put = st.sidebar.number_input("Put Strike Price", value=90.0, format="%.2f")
        quantity = st.sidebar.number_input("Contracts", value=1, step=1)

        if strangle_type == "Long Strangle":
            price_call = black_scholes(spot_price, strike_call, T, r, sigma, "call")
            price_put = black_scholes(spot_price, strike_put, T, r, sigma, "put")
            legs.append({"type": "call", "position": 1, "strike": strike_call, "premium": price_call, "quantity": quantity})
            legs.append({"type": "put", "position": 1, "strike": strike_put, "premium": price_put, "quantity": quantity})
        elif strangle_type == "Short Strangle":
            price_call = black_scholes(spot_price, strike_call, T, r, sigma, "call")
            price_put = black_scholes(spot_price, strike_put, T, r, sigma, "put")
            legs.append({"type": "call", "position": -1, "strike": strike_call, "premium": price_call, "quantity": quantity})
            legs.append({"type": "put", "position": -1, "strike": strike_put, "premium": price_put, "quantity": quantity})
    elif strategy_type == "Butterfly":
        butterfly_type = st.selectbox("Select Butterfly Type", ["Long Call Butterfly", "Short Call Butterfly",
                                                        "Long Put Butterfly", "Short Put Butterfly", "Iron Butterfly"])
        st.sidebar.header("Butterfly Spread Parameters")
        strike1 = st.sidebar.number_input("Lower Strike Price", value=90.0, format="%.2f")
        strike2 = st.sidebar.number_input("Middle Strike Price", value=100.0, format="%.2f")
        strike3 = st.sidebar.number_input("Upper Strike Price", value=110.0, format="%.2f")
        quantity = st.sidebar.number_input("Contracts", value=1, step=1)
        assert strike1 < strike2 < strike3
        assert (strike2 - spot_price)/strike2 < 0.25, "Middle strike should be close to spot price for butterfly spreads"
        if butterfly_type == "Long Call Butterfly":
            price1 = black_scholes(spot_price, strike1, T, r, sigma, "call")
            price2 = black_scholes(spot_price, strike2, T, r, sigma, "call")
            price3 = black_scholes(spot_price, strike3, T, r, sigma, "call")
            legs.append({"type": "call", "position": 1, "strike": strike1, "premium": price1, "quantity": quantity})
            legs.append({"type": "call", "position": -1, "strike": strike2, "premium": price2, "quantity": 2 * quantity})
            legs.append({"type": "call", "position": 1, "strike": strike3, "premium": price3, "quantity": quantity})
        elif butterfly_type == "Short Call Butterfly":
            price1 = black_scholes(spot_price, strike1, T, r, sigma, "call")
            price2 = black_scholes(spot_price, strike2, T, r, sigma, "call")
            price3 = black_scholes(spot_price, strike3, T, r, sigma, "call")
            legs.append({"type": "call", "position": -1, "strike": strike1, "premium": price1, "quantity": quantity})
            legs.append({"type": "call", "position": 1, "strike": strike2, "premium": price2, "quantity": 2 * quantity})
            legs.append({"type": "call", "position": -1, "strike": strike3, "premium": price3, "quantity": quantity})
        elif butterfly_type == "Long Put Butterfly":
            price1 = black_scholes(spot_price, strike1, T, r, sigma, "put")
            price2 = black_scholes(spot_price, strike2, T, r, sigma, "put")
            price3 = black_scholes(spot_price, strike3, T, r, sigma, "put")
            legs.append({"type": "put", "position": 1, "strike": strike1, "premium": price1, "quantity": quantity})
            legs.append({"type": "put", "position": -1, "strike": strike2, "premium": price2, "quantity": 2 * quantity})
            legs.append({"type": "put", "position": 1, "strike": strike3, "premium": price3, "quantity": quantity})
        elif butterfly_type == "Short Put Butterfly":
            price1 = black_scholes(spot_price, strike1, T, r, sigma, "put")
            price2 = black_scholes(spot_price, strike2, T, r, sigma, "put")
            price3 = black_scholes(spot_price, strike3, T, r, sigma, "put")
            legs.append({"type": "put", "position": -1, "strike": strike1, "premium": price1, "quantity": quantity})
            legs.append({"type": "put", "position": 1, "strike": strike2, "premium": price2, "quantity": 2 * quantity})
            legs.append({"type": "put", "position": -1, "strike": strike3, "premium": price3, "quantity": quantity})
        elif butterfly_type == "Iron Butterfly":
            price_call1 = black_scholes(spot_price, strike3, T, r, sigma, "call")
            price_call2 = black_scholes(spot_price, strike2, T, r, sigma, "call")
            price_put1 = black_scholes(spot_price, strike1, T, r, sigma, "put")
            price_put2 = black_scholes(spot_price, strike2, T, r, sigma, "put")
            legs.append({"type": "call", "position": 1, "strike": strike3, "premium": price_call1, "quantity": quantity})
            legs.append({"type": "call", "position": -1, "strike": strike2, "premium": price_call2, "quantity": quantity})
            legs.append({"type": "put", "position": 1, "strike": strike1, "premium": price_put1, "quantity": quantity})
            legs.append({"type": "put", "position": -1, "strike": strike2, "premium": price_put2, "quantity": quantity})
    elif strategy_type == "Iron Condor":
        st.sidebar.header("Iron Condor Parameters")
        strike1 = st.sidebar.number_input("Lower Strike Price (Put)", value=90.0, format="%.2f")
        strike2 = st.sidebar.number_input("Higher Strike Price (Put)", value=100.0, format="%.2f")
        strike3 = st.sidebar.number_input("Lower Strike Price (Call)", value=110.0, format="%.2f")
        strike4 = st.sidebar.number_input("Higher Strike Price (Call)", value=120.0, format="%.2f")
        quantity = st.sidebar.number_input("Contracts", value=1, step=1)
        assert strike1 < strike2 < strike3 < strike4
        price_put1 = black_scholes(spot_price, strike1, T, r, sigma, "put")
        price_put2 = black_scholes(spot_price, strike2, T, r, sigma, "put")
        price_call1 = black_scholes(spot_price, strike3, T, r, sigma, "call")
        price_call2 = black_scholes(spot_price, strike4, T, r, sigma, "call")
        legs.append({"type": "put", "position": 1, "strike": strike1, "premium": price_put1, "quantity": quantity})
        legs.append({"type": "put", "position": -1, "strike": strike2, "premium": price_put2, "quantity": quantity})
        legs.append({"type": "call", "position": -1, "strike": strike3, "premium": price_call1, "quantity": quantity})
        legs.append({"type": "call", "position": 1, "strike": strike4, "premium": price_call2, "quantity": quantity})
    plot_strategy_payoff(legs, np.linspace(0.5*spot_price, 1.5*spot_price, 200), spot_price)

elif tab == "Volatility Information":
    # -------- imports (kept local so this block is self-contained) --------
    import time, random
    import numpy as np
    import pandas as pd
    import yfinance as yf
    import plotly.graph_objects as go
    import streamlit as st

    st.header("Implied Volatility Viewer (Prefetch) — Test_IV")

    # =========================
    # Helpers (Test_IV-prefixed)
    # =========================
    RATE_LIMIT_HINTS = ("Too Many Requests", "Rate limited", "HTTP Error 429")
    COVER_FRAC = 0.35   # keep strikes covered by at least 35% of expiries
    STRIKE_RES = 160    # strike grid resolution for surface
    T_RES = 60          # maturity grid resolution for surface

    def sleep_throttle(base=0.35, jitter=0.15):
        time.sleep(max(0.05, base + random.uniform(-jitter, jitter)))

    def fetch_chain_with_retry(opt: yf.Ticker, exp: str, attempts: int = 3):
        last_err = None
        for i in range(attempts):
            try:
                return opt.option_chain(exp)
            except Exception as e:
                last_err = e
                if any(h in str(e) for h in RATE_LIMIT_HINTS):
                    time.sleep(1.2 * (2 ** i) + random.random() * 0.3)
                    continue
                raise
        raise last_err

    @st.cache_data(ttl=900)
    def load_option_chains(ticker: str, max_expiries: int = 25):
        """Return (exp_list, data_map, fetched_at) where data_map[exp] = (calls_df, puts_df)."""
        opt = yf.Ticker(ticker)
        exp_list = list(opt.options or [])[:max_expiries]
        data_map = {}
        for i, exp in enumerate(exp_list):
            chain = fetch_chain_with_retry(opt, exp)
            data_map[exp] = (chain.calls.copy(), chain.puts.copy())
            if i < len(exp_list) - 1:
                sleep_throttle()
        return exp_list, data_map, time.time()

    def clean_iv_frame(df: pd.DataFrame) -> pd.DataFrame:
        cols = ["strike", "impliedVolatility", "bid", "ask", "openInterest"]
        out = df[[c for c in cols if c in df.columns]].copy()
        out = out.dropna(subset=["strike", "impliedVolatility"])
        if "bid" in out.columns and "ask" in out.columns:
            out = out[(out["bid"] > 0) & (out["ask"] > 0)]
        if "openInterest" in out.columns:
            out = out[out["openInterest"].fillna(0) > 0]
        out["impliedVolatility"] = out["impliedVolatility"].clip(0.01, 2.00)  # 1%–200%
        return out

    def nan_smooth(A: np.ndarray, axis: int) -> np.ndarray:
        # mild 3-tap smoothing; NaN-aware
        k = np.array([0.25, 0.5, 0.25])
        if axis == 0:
            pad = ((1,1),(0,0))
            Ap = np.pad(np.nan_to_num(A, nan=0.0), pad, mode="edge")
            Wp = np.pad(np.isfinite(A).astype(float), pad, mode="edge")
            num = k[0]*Ap[:-2,:] + k[1]*Ap[1:-1,:] + k[2]*Ap[2:,:]
            den = k[0]*Wp[:-2,:] + k[1]*Wp[1:-1,:] + k[2]*Wp[2:,:]
        else:
            pad = ((0,0),(1,1))
            Ap = np.pad(np.nan_to_num(A, nan=0.0), pad, mode="edge")
            Wp = np.pad(np.isfinite(A).astype(float), pad, mode="edge")
            num = k[0]*Ap[:,:-2] + k[1]*Ap[:,1:-1] + k[2]*Ap[:,2:]
            den = k[0]*Wp[:,:-2] + k[1]*Wp[:,1:-1] + k[2]*Wp[:,2:]
        out = np.divide(num, den, out=np.full_like(num, np.nan), where=den>0)
        return out

    # =========================
    # Sidebar (unique names)
    # =========================
    ticker = st.sidebar.text_input("Ticker (Test_IV)", value="AAPL").upper()
    max_expiries = int(st.sidebar.number_input("Prefetch expiries (Test_IV)", 1, 60, 25))
    refresh_btn = st.sidebar.button("Refresh cached data (Test_IV)")

    if refresh_btn:
        try:
            load_option_chains.clear()
        except Exception:
            st.cache_data.clear()
        st.experimental_rerun()

    # =========================
    # Prefetch & cache
    # =========================
    with st.spinner(f"Fetching up to {max_expiries} expiries for {ticker} ..."):
        try:
            expiries_list, chains_by_expiry, fetched_at = load_option_chains(
                ticker, max_expiries
            )
        except Exception as e:
            st.error(f"[Test_IV] Error prefetching: {e}")
            st.stop()

    if not expiries_list:
        st.warning("[Test_IV] No options data available for this ticker.")
        st.stop()

    expiry_choice = st.sidebar.selectbox("Select Expiration (Test_IV)", expiries_list)

    calls_df, puts_df = chains_by_expiry.get(expiry_choice, (None, None))
    if calls_df is None or puts_df is None or (calls_df.empty and puts_df.empty):
        st.warning("[Test_IV] No data for this expiry.")
        st.stop()

    # =========================
    # 2D smiles (explicit colors)
    # =========================
    calls_df = calls_df.dropna(subset=["strike", "impliedVolatility"])
    puts_df  = puts_df.dropna(subset=["strike", "impliedVolatility"])

    col_left, col_right = st.columns(2)

    with col_left:
        fig_calls_2d = go.Figure()
        fig_calls_2d.add_trace(go.Scatter(
            x=calls_df["strike"],
            y=calls_df["impliedVolatility"] * 100.0,
            mode="lines+markers",
            name="Calls (Test_IV)",
            line=dict(color="royalblue"),
            marker=dict(size=6, color="royalblue")
        ))
        fig_calls_2d.update_layout(
            title=f"Call IV Smile — {expiry_choice} (Test_IV)",
            xaxis_title="Strike",
            yaxis_title="IV (%)",
            hovermode="x unified",
            template="plotly_dark",
        )
        st.plotly_chart(fig_calls_2d, use_container_width=True)

    with col_right:
        fig_puts_2d = go.Figure()
        fig_puts_2d.add_trace(go.Scatter(
            x=puts_df["strike"],
            y=puts_df["impliedVolatility"] * 100.0,
            mode="lines+markers",
            name="Puts (Test_IV)",
            line=dict(color="darkorange"),
            marker=dict(size=6, color="darkorange")
        ))
        fig_puts_2d.update_layout(
            title=f"Put IV Smile — {expiry_choice} (Test_IV)",
            xaxis_title="Strike",
            yaxis_title="IV (%)",
            hovermode="x unified",
            template="plotly_dark",
        )
        st.plotly_chart(fig_puts_2d, use_container_width=True)

    # =========================
    # 3D curves (per selected expiry)
    # =========================
    calls_sorted = calls_df.sort_values("strike")
    puts_sorted  = puts_df.sort_values("strike")

    call_x = calls_sorted["strike"].to_numpy()
    call_y = ((calls_sorted["bid"].fillna(0) + calls_sorted["ask"].fillna(0)) / 2.0).to_numpy()
    call_z = (calls_sorted["impliedVolatility"] * 100.0).to_numpy()

    put_x = puts_sorted["strike"].to_numpy()
    put_y = ((puts_sorted["bid"].fillna(0) + puts_sorted["ask"].fillna(0)) / 2.0).to_numpy()
    put_z = (puts_sorted["impliedVolatility"] * 100.0).to_numpy()

    col3, col4 = st.columns([1, 1])
    with col3:
        fig_calls_3d = go.Figure()
        fig_calls_3d.add_trace(go.Scatter3d(
            x=call_x, y=call_y, z=call_z,
            mode="lines+markers",
            line=dict(color="royalblue", width=4),
            marker=dict(size=5, color=call_z, colorscale="Viridis", opacity=0.9),
            name="Calls 3D (Test_IV)"
        ))
        fig_calls_3d.update_layout(
            title=f"Call 3D IV Curve — {expiry_choice}",
            scene=dict(
                xaxis_title="Strike",
                yaxis_title="Option Price ($)",
                zaxis_title="IV (%)",
                camera=dict(eye=dict(x=2, y=-1.8, z=1)),
                aspectmode="manual",
                aspectratio=dict(x=1.5, y=1.0, z=1.0),
            ),
            template="plotly_dark",
            margin=dict(l=0, r=0, b=0, t=40),
            height=700,
        )
        st.plotly_chart(fig_calls_3d, use_container_width=True)

    with col4:
        fig_puts_3d = go.Figure()
        fig_puts_3d.add_trace(go.Scatter3d(
            x=put_x, y=put_y, z=put_z,
            mode="lines+markers",
            line=dict(color="darkorange", width=4),
            marker=dict(size=5, color=put_z, colorscale="Plasma", opacity=0.9),
            name="Puts 3D (Test_IV)"
        ))
        fig_puts_3d.update_layout(
            title=f"Put 3D IV Curve — {expiry_choice}",
            scene=dict(
                xaxis_title="Strike",
                yaxis_title="Option Price ($)",
                zaxis_title="IV (%)",
                camera=dict(eye=dict(x=2, y=-1.8, z=1)),
                aspectmode="manual",
                aspectratio=dict(x=1.5, y=1.0, z=1.0),
            ),
            template="plotly_dark",
            margin=dict(l=0, r=0, b=0, t=40),
            height=700,
        )
        st.plotly_chart(fig_puts_3d, use_container_width=True)

    st.caption("Test_IV uses cached multi-expiry chains. Switching expiries doesn’t re-download.")

    # ==========================================================
    # DUAL IV SURFACES — Calls vs Puts (Y = Maturity in years)
    # ==========================================================
    st.subheader("IV Surfaces — Calls vs Puts (Maturity × Strike)")

    def build_surface_for_side(side: str):
        """
        side: 'call' or 'put'
        Returns (x_grid, T_grid, Z_T) or (None, None, None) if not enough data.
        Uses coverage-threshold strike grid and mild smoothing in both directions.
        """
        # 1) Per-expiry curves for the chosen side
        curves = []
        valid_exps = []
        for exp in expiries_list:
            c_df, p_df = chains_by_expiry.get(exp, (None, None))
            df = (c_df if side == "call" else p_df)
            if df is None or df.empty:
                continue
            df = clean_iv_frame(df)
            if df.empty:
                continue

            x = df["strike"].to_numpy()
            z = (df["impliedVolatility"].to_numpy() * 100.0)  # %
            m = np.isfinite(x) & np.isfinite(z)
            x, z = x[m], z[m]
            if len(x) < 2:
                continue

            # tiny median smooth along strikes
            z = pd.Series(z).rolling(window=3, center=True, min_periods=1).median().to_numpy()

            curves.append((x, z))
            valid_exps.append(exp)

        if len(valid_exps) < 2:
            return None, None, None

        # 2) Coverage-threshold strike grid (keeps X wider but not holey)
        union = np.unique(np.concatenate([x for x, _ in curves]))
        x_grid = np.linspace(union.min(), union.max(), STRIKE_RES) if len(union) >= 4 else union

        coverage = np.zeros_like(x_grid, dtype=int)
        for x, _ in curves:
            lo, hi = x.min(), x.max()
            coverage += (x_grid >= lo) & (x_grid <= hi)
        min_cover = max(1, int(np.ceil(COVER_FRAC * len(curves))))
        keep = coverage >= min_cover
        x_grid = x_grid[keep]

        if len(x_grid) < 5:
            # fallback to strict intersection if threshold collapses
            lo = max(x.min() for x, _ in curves)
            hi = min(x.max() for x, _ in curves)
            if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
                return None, None, None
            x_grid = np.linspace(lo, hi, max(5, STRIKE_RES // 2))

        # 3) Interpolate each expiry onto strike grid
        Z_by_exp = np.empty((len(curves), len(x_grid)))
        for i, (x, z) in enumerate(curves):
            o = np.argsort(x)
            Z_by_exp[i, :] = np.interp(x_grid, x[o], z[o])

        # 4) Maturity (years), sorted
        now_utc = pd.Timestamp.utcnow()
        T_years = []
        for e in valid_exps:
            ets = pd.to_datetime(e, utc=True, errors="coerce")
            if pd.isna(ets):
                T_years.append(np.nan)
            else:
                dt_days = (ets - now_utc).total_seconds() / 86400.0
                T_years.append(max(0.0, dt_days / 365.25))
        T_years = np.asarray(T_years)
        ordT = np.argsort(T_years)
        T_years = T_years[ordT]
        Z_by_exp = Z_by_exp[ordT, :]

        # 5) Interpolate across maturity to regular grid
        T_grid = np.linspace(float(np.nanmin(T_years)), float(np.nanmax(T_years)), T_RES)
        Z_T = np.empty((len(T_grid), len(x_grid)))
        for j in range(len(x_grid)):
            col = Z_by_exp[:, j]
            valid = np.isfinite(col) & np.isfinite(T_years)
            if valid.sum() >= 2:
                Z_T[:, j] = np.interp(T_grid, T_years[valid], col[valid])
            elif valid.sum() == 1:
                Z_T[:, j] = float(col[valid][0])
            else:
                Z_T[:, j] = np.nan

        # 6) Mild smoothing along strike then maturity
        Z_T = nan_smooth(nan_smooth(Z_T, axis=1), axis=0)

        return x_grid, T_grid, Z_T

    # Build both surfaces first so we can share a z-axis range (easier comparison)
    xC, TC, ZC = build_surface_for_side("call")
    xP, TP, ZP = build_surface_for_side("put")

    if xC is None and xP is None:
        st.info("[Test_IV] Not enough data to draw call or put surfaces.")
    else:
        # Determine a common Z range if both exist
        zmin = np.inf
        zmax = -np.inf
        if ZC is not None:
            zmin = min(zmin, np.nanmin(ZC))
            zmax = max(zmax, np.nanmax(ZC))
        if ZP is not None:
            zmin = min(zmin, np.nanmin(ZP))
            zmax = max(zmax, np.nanmax(ZP))
        if not np.isfinite(zmin) or not np.isfinite(zmax) or zmin >= zmax:
            zmin, zmax = 0, 200

        col_calls, col_puts = st.columns(2)

        if xC is not None:
            with col_calls:
                figC = go.Figure(data=[
                    go.Surface(
                        x=xC, y=TC, z=ZC,
                        colorscale="Viridis", showscale=True
                    )
                ])
                figC.update_layout(
                    title="Call IV Surface — Maturity (years) × Strike",
                    scene=dict(
                        xaxis_title="Strike",
                        yaxis_title="Maturity (years)",
                        zaxis_title="IV (%)",
                        zaxis=dict(range=[float(zmin), float(zmax)]),
                        yaxis=dict(tickformat=".2f"),
                        camera=dict(eye=dict(x=1.6, y=1.2, z=1.0)),
                        aspectmode="manual",
                        aspectratio=dict(x=1.6, y=1.0, z=0.8),
                    ),
                    template="plotly_dark",
                    height=700,
                    margin=dict(l=0, r=0, b=0, t=40),
                )
                st.plotly_chart(figC, use_container_width=True)

        if xP is not None:
            with col_puts:
                figP = go.Figure(data=[
                    go.Surface(
                        x=xP, y=TP, z=ZP,
                        colorscale="Plasma", showscale=True
                    )
                ])
                figP.update_layout(
                    title="Put IV Surface — Maturity (years) × Strike",
                    scene=dict(
                        xaxis_title="Strike",
                        yaxis_title="Maturity (years)",
                        zaxis_title="IV (%)",
                        zaxis=dict(range=[float(zmin), float(zmax)]),
                        yaxis=dict(tickformat=".2f"),
                        camera=dict(eye=dict(x=1.6, y=1.2, z=1.0)),
                        aspectmode="manual",
                        aspectratio=dict(x=1.6, y=1.0, z=0.8),
                    ),
                    template="plotly_dark",
                    height=700,
                    margin=dict(l=0, r=0, b=0, t=40),
                )
                st.plotly_chart(figP, use_container_width=True)

    st.caption(
        "Surfaces use calls-only and puts-only data separately. "
        "In theory, call and put IVs match at the same (K, T); differences here mainly reflect quote noise and liquidity."
    )

    # ==============================================
    # RAW IV SCATTER — Calls vs Puts (no interpolation)
    # ==============================================
    st.subheader("IV Scatter — Maturity (years) × Strike (Calls vs Puts)")

    def build_scatter_side(side: str):
        """Return (X_strike, Y_maturity_years, Z_iv_pct) for the chosen side ('call'|'put')."""
        X, T, Z = [], [], []
        now_utc = pd.Timestamp.utcnow()
        for exp in expiries_list:
            c_df, p_df = chains_by_expiry.get(exp, (None, None))
            df = c_df if side == "call" else p_df
            if df is None or df.empty:
                continue
            df = clean_iv_frame(df)
            if df.empty:
                continue

            ets = pd.to_datetime(exp, utc=True, errors="coerce")
            if pd.isna(ets):
                continue
            T_years = max(0.0, (ets - now_utc).total_seconds() / 86400.0 / 365.25)

            x = df["strike"].to_numpy()
            z = (df["impliedVolatility"].to_numpy() * 100.0)  # %
            m = np.isfinite(x) & np.isfinite(z)
            if m.sum() == 0:
                continue

            X.append(x[m])
            Z.append(z[m])
            T.append(np.full(m.sum(), T_years))

        if len(X) == 0:
            return None, None, None
        return np.concatenate(X), np.concatenate(T), np.concatenate(Z)

    # Build scatter datasets
    sxC, sT_C, sZ_C = build_scatter_side("call")
    sxP, sT_P, sZ_P = build_scatter_side("put")

    if sxC is None and sxP is None:
        st.info("[Test_IV] Not enough data to draw scatter plots.")
    else:
        # Share a Z-range for easier visual comparison (include surfaces if present)
        zmin = np.inf
        zmax = -np.inf
        for arr in (sZ_C, sZ_P):
            if arr is not None and arr.size:
                zmin = min(zmin, float(np.nanmin(arr)))
                zmax = max(zmax, float(np.nanmax(arr)))
        # Try to fold in surface ranges if those vars exist
        try:
            for arr in (ZC, ZP):
                if arr is not None and np.size(arr):
                    zmin = min(zmin, float(np.nanmin(arr)))
                    zmax = max(zmax, float(np.nanmax(arr)))
        except NameError:
            pass
        if not np.isfinite(zmin) or not np.isfinite(zmax) or zmin >= zmax:
            zmin, zmax = 0.0, 200.0

        col_sc1, col_sc2 = st.columns(2)

        if sxC is not None:
            with col_sc1:
                fig_scC = go.Figure(go.Scatter3d(
                    x=sxC, y=sT_C, z=sZ_C,
                    mode="markers",
                    marker=dict(size=3, opacity=0.7, color=sZ_C, colorscale="Viridis"),
                    name="Calls (scatter)"
                ))
                fig_scC.update_layout(
                    title="Call IV Scatter — Maturity × Strike",
                    scene=dict(
                        xaxis_title="Strike",
                        yaxis_title="Maturity (years)",
                        zaxis_title="IV (%)",
                        zaxis=dict(range=[float(zmin), float(zmax)]),
                        yaxis=dict(tickformat=".2f"),
                        camera=dict(eye=dict(x=1.6, y=1.2, z=1.0)),
                        aspectmode="manual",
                        aspectratio=dict(x=1.6, y=1.0, z=0.8),
                    ),
                    template="plotly_dark",
                    height=700,
                    margin=dict(l=0, r=0, b=0, t=40),
                )
                st.plotly_chart(fig_scC, use_container_width=True)

        if sxP is not None:
            with col_sc2:
                fig_scP = go.Figure(go.Scatter3d(
                    x=sxP, y=sT_P, z=sZ_P,
                    mode="markers",
                    marker=dict(size=3, opacity=0.7, color=sZ_P, colorscale="Plasma"),
                    name="Puts (scatter)"
                ))
                fig_scP.update_layout(
                    title="Put IV Scatter — Maturity × Strike",
                    scene=dict(
                        xaxis_title="Strike",
                        yaxis_title="Maturity (years)",
                        zaxis_title="IV (%)",
                        zaxis=dict(range=[float(zmin), float(zmax)]),
                        yaxis=dict(tickformat=".2f"),
                        camera=dict(eye=dict(x=1.6, y=1.2, z=1.0)),
                        aspectmode="manual",
                        aspectratio=dict(x=1.6, y=1.0, z=0.8),
                    ),
                    template="plotly_dark",
                    height=700,
                    margin=dict(l=0, r=0, b=0, t=40),
                )
                st.plotly_chart(fig_scP, use_container_width=True)

    st.caption("Scatter shows raw, cleaned quotes (no interpolation); compare it to the smoothed surfaces above.")
