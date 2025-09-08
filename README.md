# Options Analysis Toolkit (Streamlit)

An interactive web application for exploring options pricing, strategies, Greeks, and implied volatility surfaces.  
The project separates financial math functions from the user interface for clarity, maintainability, and reuse.

---

## Features

- **Black–Scholes Pricer**  
  Compute European call and put prices with user-controlled parameters.

- **Greeks Calculations**  
  Delta, Gamma, Vega, Theta, Rho.

- **Implied Volatility Solver**  
  Inverts the Black–Scholes model numerically to estimate volatility.

- **Implied Volatility Analysis**  
  - IV smiles  
  - IV surfaces  

- **3D Surfaces (Price and Greeks)**  
  - Option price surfaces as a function of spot × volatility or spot × time  
  - Greek surfaces (Delta, Gamma, Vega, Theta, Rho) across parameter grids  

- **Heatmaps (Price and Greeks)**  
  - Option price heatmaps for different input ranges  
  - Greek heatmaps to visualize sensitivity to spot price, volatility, and time  

- **Strategy Payoffs**  
  Visualize single options and spreads with P&L charts.

- **Performance Optimizations**  
  - `@st.cache_data` for data pulls and heavy computations  
  - JAX for automatic differentiation and speedups  

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/kgruz/Options-Analysis-Toolkit/
   cd my-project
   ```
2. Install Dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Alternatively:
   - You can access this project at https://options-analysis-toolkit.streamlit.app/
   - You can also download the file from my GitHub repository.

---

## Usage
1. Run the Streamlit app:
    ```bash
    streamlit run main.py
    ```
2. Navigate to the provided URL.
3. Try out the tools!

---

## Screenshots

---

## Technology Stack

- [Streamlit](https://streamlit.io/) — interactive UI framework
- [NumPy](https://numpy.org/) / [SciPy](https://scipy.org/) — numerical computing
- [Pandas](https://pandas.pydata.org/) — data manipulation
- [Matplotlib](https://matplotlib.org/) — Plotting
- [Plotly](https://plotly.com/python/) — interactive 3D visualizations
- [yfinance](https://pypi.org/project/yfinance/) — options chain data
- [JAX](https://jax.readthedocs.io/) — fast numerical computing and gradients

