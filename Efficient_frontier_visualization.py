import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import cvxpy as cp
import plotly.express as px
from datetime import datetime

def download_nse50_data(start_date, end_date, analysis_type):
    
    ticker_list = [
    'ABB.NS', 'ABCAPITAL.NS', 'ABFRL.NS', 'ACC.NS', 'ADANIENT.NS', 'ADANIGREEN.NS', 'ADANIPORTS.NS', 'ADANIPOWER.NS',
    'AFFLE.NS', 'AJANTPHARM.NS', 'ALKEM.NS', 'AMBUJACEM.NS', 'APLAPOLLO.NS',
    'APOLLOHOSP.NS', 'APOLLOTYRE.NS', 'ASHOKLEY.NS', 'ASIANPAINT.NS', 'ASTRAL.NS', 'AUROPHARMA.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS',
    'BAJAJFINSV.NS', 'BAJFINANCE.NS', 'BALKRISIND.NS', 'BALRAMCHIN.NS', 'BANDHANBNK.NS', 'BANKBARODA.NS', 'BATAINDIA.NS', 'BERGEPAINT.NS',
    'BHARATFORG.NS', 'BHARTIARTL.NS', 'BHEL.NS', 'BIOCON.NS', 'BOSCHLTD.NS', 'BPCL.NS', 'BRITANNIA.NS', 'BSOFT.NS', 'CANBK.NS',
    'CANFINHOME.NS', 'CAPLIPOINT.NS', 'CASTROLIND.NS', 'CDSL.NS', 'CENTRALBK.NS', 'CENTURYTEX.NS', 'CIPLA.NS', 'COALINDIA.NS',
    'COFORGE.NS', 'COLPAL.NS', 'CONCOR.NS', 'COROMANDEL.NS', 'CROMPTON.NS', 'CUB.NS', 'CUMMINSIND.NS', 'DABUR.NS', 'DALBHARAT.NS',
    'DEEPAKNTR.NS', 'DELHIVERY.NS', 'DELTACORP.NS', 'DIVISLAB.NS', 'DIXON.NS', 'DLF.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'EMAMILTD.NS',
    'ESCORTS.NS', 'EXIDEIND.NS', 'FEDERALBNK.NS', 'FINCABLES.NS', 'FORTIS.NS', 'GAIL.NS', 'GLAND.NS', 'GLENMARK.NS', 'GMRINFRA.NS',
    'GNFC.NS', 'GODREJAGRO.NS', 'GODREJCP.NS', 'GODREJIND.NS', 'GRANULES.NS', 'GRASIM.NS', 'GSPL.NS', 'GUJGASLTD.NS', 'HAVELLS.NS',
    'HCLTECH.NS', 'HDFCAMC.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDCOPPER.NS', 'HINDPETRO.NS',
    'HINDUNILVR.NS', 'ICICIBANK.NS', 'ICICIGI.NS', 'ICICIPRULI.NS', 'IDEA.NS', 'IDFCFIRSTB.NS', 'IEX.NS', 'IGL.NS',
    'INDHOTEL.NS', 'INDIAMART.NS', 'INDIGO.NS', 'INDUSINDBK.NS', 'INDUSTOWER.NS', 'INFY.NS', 'INTELLECT.NS', 'IOC.NS', 'IRCTC.NS',
    'ITC.NS', 'JINDALSTEL.NS', 'JSWENERGY.NS', 'JSWSTEEL.NS', 'JUBLFOOD.NS', 'KOTAKBANK.NS', 'LALPATHLAB.NS', 'LAURUSLABS.NS',
    'LICHSGFIN.NS', 'LT.NS', 'LTIM.NS', 'LTTS.NS', 'LUPIN.NS', 'M&M.NS', 'M&MFIN.NS', 'MANAPPURAM.NS', 'MARICO.NS', 'MARUTI.NS',
    'MCX.NS', 'METROPOLIS.NS', 'MFSL.NS', 'MGL.NS', 'MOTHERSON.NS', 'MPHASIS.NS', 'MRF.NS', 'MUTHOOTFIN.NS',
    'NAM-INDIA.NS', 'NBCC.NS', 'NESTLEIND.NS', 'NMDC.NS', 'NTPC.NS', 'OBEROIRLTY.NS', 'ONGC.NS', 'PAGEIND.NS', 'PAYTM.NS', 'PEL.NS',
    'PETRONET.NS', 'PFC.NS', 'PFIZER.NS', 'PIIND.NS', 'PNB.NS', 'POLYCAB.NS', 'POWERGRID.NS', 'PVRINOX.NS', 'RAIN.NS', 'RBLBANK.NS',
    'RECLTD.NS', 'RELIANCE.NS', 'SAIL.NS', 'SBICARD.NS', 'SBILIFE.NS', 'SBIN.NS', 'SHREECEM.NS', 'SIEMENS.NS', 'SRF.NS',
    'STAR.NS', 'SUNPHARMA.NS', 'SUNTV.NS', 'SYNGENE.NS', 'TATACHEM.NS', 'TATACOMM.NS', 'TATACONSUM.NS', 'TATAELXSI.NS', 'TATAMOTORS.NS',
    'TATAPOWER.NS', 'TATASTEEL.NS', 'TCS.NS', 'TECHM.NS', 'THERMAX.NS', 'TIINDIA.NS', 'TORNTPHARM.NS', 'TORNTPOWER.NS', 'TRENT.NS',
    'TVSMOTOR.NS', 'UBL.NS', 'ULTRACEMCO.NS', 'UNIONBANK.NS', 'UPL.NS', 'VBL.NS', 'VEDL.NS', 'VOLTAS.NS', 'WHIRLPOOL.NS', 'WIPRO.NS',
    'YESBANK.NS', 'ZEEL.NS', 'ZOMATO.NS'
]
    data = yf.download(ticker_list, start=start_date, end=end_date)
    data = data.dropna(axis=1, how='all')
    
    # Calculate returns
    if analysis_type == 'Daily return':
        returns = data['Close'].pct_change().dropna()
    elif analysis_type == 'Night return':
        returns = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
    else:
        returns = (data['Close'] - data['Open']) / data['Open']

    returns.dropna(inplace=True)
    return returns

def optimize_portfolio(returns, target_return):
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(mean_returns)
    
    weights = cp.Variable(num_assets, nonneg=True)
    portfolio_variance = cp.quad_form(weights, cov_matrix)
    constraints = [cp.sum(weights) == 1, weights @ mean_returns == target_return]
    problem = cp.Problem(cp.Minimize(portfolio_variance), constraints)
    problem.solve()
    
    return weights.value, weights.value @ mean_returns, problem.value

st.sidebar.header("Efficient Frontier visualization with NSE 200 assets")
with st.sidebar.form("Segmentation"):
    st.write("Note: Day return is the return during trading hours.\n Night return is the return during previous night.\n Daily return is combination of day and night return")
    start_date = st.date_input("Select start date", datetime(2023, 1, 1))
    end_date = st.date_input("Select end date", datetime(2024, 12, 31))
    analysis_type = st.selectbox("Choose Return Type", ['Day return', 'Night return', 'Daily return'])
    button = st.form_submit_button("Apply Changes")
# start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
# end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))

st.write("Note: This visualization plots Daily mean percent return vs risk as the varience of percent returns.")
st.write("Standard daviation of percent returns are available on hovering on the data points.")
st.write("Use zoom, pan tools to adjust visualization.")


if button or "form_submitted" not in st.session_state:
    
    data = download_nse50_data(start_date, end_date, analysis_type)
    # st.write("Downloaded stock data:", data.tail())

    solutions = []
    target_returns = np.linspace(0.00, 0.005, 30)
    for target in target_returns:
        try:
            weights, mean_ret, variance = optimize_portfolio(data, target)
            solutions.append((mean_ret * 100, variance))  # Convert mean return to percent
        except:
            continue

    solutions = np.array(solutions)
    print(solutions)
    individual_assets = pd.DataFrame({
        "Stock": data.columns,
        "Mean Return (%)": data.mean() * 100,
        "Variance": data.var(),
        "Std Dev (%)": np.sqrt(data.var()) * 100
    })

    fig = px.line(
        x=solutions[:, 1], y=solutions[:, 0], markers=True,
        labels={"x": "Risk (Variance of percent return)", "y": "Mean Return (%)"}, title="Efficient Frontier",
        hover_data={"Standard deviation(%)":np.round(np.sqrt(solutions[:, 1])*100, 2)}
    )
    fig.add_scatter(
        x=individual_assets["Variance"], y=individual_assets["Mean Return (%)"],
        mode='markers+text', text=individual_assets["Stock"], textposition="top center",
        marker=dict(size=8), name="Individual Assets",
        hovertext=individual_assets.apply(lambda row: f"{row['Stock']}: Std Dev {row['Std Dev (%)']:.2f}%", axis=1)
    )
    fig.update_layout(height = 600, width = 600)
    st.plotly_chart(fig)
