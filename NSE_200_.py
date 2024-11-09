import yfinance as yf
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime

# Streamlit app title and sidebar inputs
st.title("Mean Night vs. Day Return for Stocks")

# Sidebar for date selection
start_date = st.sidebar.date_input("Select start date", datetime(2024, 1, 1))
end_date = st.sidebar.date_input("Select end date", datetime(2024, 12, 31))

# List of stocks to analyze
# stocks = ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'NVDA', 'META', 'NFLX', 'SPCE']  # Example list of stocks
stocks = [
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

def download_data(stocks, start_date, end_date):
    # Download all stocks' data at once
    data = yf.download(stocks, start=start_date, end=end_date)[['Open', 'Close']]
    return data

def calculate_returns(data):
    # Calculate mean night and day returns for each stock
    stock_data = {}
    for stock in data.columns.levels[1]:  # Iterate over each stock
        df = data.xs(stock, level=1, axis=1).dropna()  # Get data for this stock
        df['Previous Close'] = df['Close'].shift(1)
        df['Night Return'] = (df['Open'] - df['Previous Close']) * 100 / df['Previous Close']
        df['Day Return'] = (df['Close'] - df['Open']) * 100 / df['Open']
        
        # Store mean night and day returns
        stock_data[stock] = {
            'Mean Night Return': df['Night Return'].mean(),
            'Mean Day Return': df['Day Return'].mean()
        }
        
    return pd.DataFrame(stock_data).T

def plot_returns(returns_df):
    # Create scatter plot using Plotly
    fig = px.scatter(
        returns_df,
        x='Mean Night Return',
        y='Mean Day Return',
        text=returns_df.index,
        labels={'x': 'Mean Night Return (%)', 'y': 'Mean Day Return (%)'},
        title='Mean Night vs. Day Return for Stocks'
    )
    fig.update_traces(textposition='top center')
    return fig

# Download data and calculate returns
if start_date < end_date:
    data = download_data(stocks, start_date, end_date)
    returns_df = calculate_returns(data)

    # Plot the returns
    fig = plot_returns(returns_df)
    # Update the layout to adjust text position and height of the figure
    fig.update_layout(
        height=800,  # Set height of the plot
        title_font_size=20,  # Adjust the title font size (optional)
    )
    st.plotly_chart(fig)
else:
    st.error("Please ensure that the start date is earlier than the end date.")
