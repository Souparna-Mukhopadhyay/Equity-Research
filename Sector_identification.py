import streamlit as st
from datetime import datetime 
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# Sidebar inputs for stock selection, date range, and rolling period
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

start_date = st.sidebar.date_input("Select start date", datetime(2024, 1, 1))
end_date = st.sidebar.date_input("Select end date", datetime(2024, 12, 31))
analysis_type = st.sidebar.selectbox("Choose Return Type", ['Daily', 'Daily Night', 'Daily Day'])
perplexity = st.sidebar.number_input("Perplexity", min_value=1, max_value=60, value=8)

data = yf.download(stocks, start=start_date, end=end_date)

if analysis_type == 'Daily':
    returns = data['Close'].pct_change().dropna()
elif analysis_type == 'Daily Night':
    returns = (data['Open']-data['Close'].shift(1))/data['Close'].shift(1)
else:
    returns = (data['Close']-data['Open'])/data['Open']
    
returns.dropna(inplace=True)

correlation_matrix = returns.corr()

# Create a distance matrix from the correlation matrix (1 - correlation)
distance_matrix = 1 - correlation_matrix

# Apply t-SNE to reduce dimensionality (n_components=2 for 2D visualization)
tsne = TSNE(n_components=2, metric='precomputed',init = 'random', random_state=41, perplexity=perplexity,)
tsne_results = tsne.fit_transform(distance_matrix)

# Create a DataFrame for the t-SNE results
tsne_df = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'], index=correlation_matrix.index)

# Add the tickers (or labels) as a column for hover information
tsne_df['Ticker'] = tsne_df.index


# Create an interactive Plotly scatter plot
fig = px.scatter(tsne_df, x='TSNE1', y='TSNE2', text='Ticker',
                 hover_name='Ticker', title="t-SNE Visualization Based on Correlation Matrix")

# Update the layout to adjust text position and height of the figure
fig.update_layout(
    height=800,  # Set height of the plot
    title_font_size=20,  # Adjust the title font size (optional)
)

# Update the text position to be more readable (optional)
fig.update_traces(textposition='top center')

# Show the interactive plot
st.plotly_chart(fig)