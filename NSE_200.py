import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import jarque_bera, skew, kurtosis
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import boxcox
from sklearn.metrics import r2_score
from datetime import datetime

# Function to calculate Heikin-Ashi close prices
def heikin_ashi(df):
    ha_df = df.copy()
    ha_df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    return ha_df

# List of NIFTY 50 tickers
# nifty50_tickers = ['7203.T', '6758.T', '9432.T', '8306.T', '9984.T', '6861.T', '8058.T', '8015.T', '4502.T', '9983.T',
#     'BABA', 'TCEHY', 'JD', 'PDD', 'BIDU', 'NIO', 'LI', 'XPEV', 'NTES', 'BILI',
#     'AAPL', 'GOOGL', 'TSLA', 'MSFT', 'NVDA', 'META', 'NFLX',
#     'ABB.NS', 'ABCAPITAL.NS', 'ABFRL.NS', 'ACC.NS', 'ADANIENT.NS', 'ADANIGREEN.NS', 'ADANIPORTS.NS', 'ADANIPOWER.NS',
#     'AFFLE.NS', 'AJANTPHARM.NS', 'ALKEM.NS', 'AMBUJACEM.NS', 'APLAPOLLO.NS',
#     'APOLLOHOSP.NS', 'APOLLOTYRE.NS', 'ASHOKLEY.NS', 'ASIANPAINT.NS', 'ASTRAL.NS', 'AUROPHARMA.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS',
#     'BAJAJFINSV.NS', 'BAJFINANCE.NS', 'BALKRISIND.NS', 'BALRAMCHIN.NS', 'BANDHANBNK.NS', 'BANKBARODA.NS', 'BATAINDIA.NS', 'BERGEPAINT.NS',
#     'BHARATFORG.NS', 'BHARTIARTL.NS', 'BHEL.NS', 'BIOCON.NS', 'BOSCHLTD.NS', 'BPCL.NS', 'BRITANNIA.NS', 'BSOFT.NS', 'CANBK.NS',
#     'CANFINHOME.NS', 'CAPLIPOINT.NS', 'CASTROLIND.NS', 'CDSL.NS', 'CENTRALBK.NS', 'CENTURYTEX.NS', 'CIPLA.NS', 'COALINDIA.NS',
#     'COFORGE.NS', 'COLPAL.NS', 'CONCOR.NS', 'COROMANDEL.NS', 'CROMPTON.NS', 'CUB.NS', 'CUMMINSIND.NS', 'DABUR.NS', 'DALBHARAT.NS',
#     'DEEPAKNTR.NS', 'DELHIVERY.NS', 'DELTACORP.NS', 'DIVISLAB.NS', 'DIXON.NS', 'DLF.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'EMAMILTD.NS',
#     'ESCORTS.NS', 'EXIDEIND.NS', 'FEDERALBNK.NS', 'FINCABLES.NS', 'FORTIS.NS', 'GAIL.NS', 'GLAND.NS', 'GLENMARK.NS', 'GMRINFRA.NS',
#     'GNFC.NS', 'GODREJAGRO.NS', 'GODREJCP.NS', 'GODREJIND.NS', 'GRANULES.NS', 'GRASIM.NS', 'GSPL.NS', 'GUJGASLTD.NS', 'HAVELLS.NS',
#     'HCLTECH.NS', 'HDFCAMC.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDCOPPER.NS', 'HINDPETRO.NS',
#     'HINDUNILVR.NS', 'ICICIBANK.NS', 'ICICIGI.NS', 'ICICIPRULI.NS', 'IDEA.NS', 'IDFCFIRSTB.NS', 'IEX.NS', 'IGL.NS',
#     'INDHOTEL.NS', 'INDIAMART.NS', 'INDIGO.NS', 'INDUSINDBK.NS', 'INDUSTOWER.NS', 'INFY.NS', 'INTELLECT.NS', 'IOC.NS', 'IRCTC.NS',
#     'ITC.NS', 'JINDALSTEL.NS', 'JSWENERGY.NS', 'JSWSTEEL.NS', 'JUBLFOOD.NS', 'KOTAKBANK.NS', 'LALPATHLAB.NS', 'LAURUSLABS.NS',
#     'LICHSGFIN.NS', 'LT.NS', 'LTIM.NS', 'LTTS.NS', 'LUPIN.NS', 'M&M.NS', 'M&MFIN.NS', 'MANAPPURAM.NS', 'MARICO.NS', 'MARUTI.NS',
#     'MCX.NS', 'METROPOLIS.NS', 'MFSL.NS', 'MGL.NS', 'MOTHERSON.NS', 'MPHASIS.NS', 'MRF.NS', 'MUTHOOTFIN.NS',
#     'NAM-INDIA.NS', 'NBCC.NS', 'NESTLEIND.NS', 'NMDC.NS', 'NTPC.NS', 'OBEROIRLTY.NS', 'ONGC.NS', 'PAGEIND.NS', 'PAYTM.NS', 'PEL.NS',
#     'PETRONET.NS', 'PFC.NS', 'PFIZER.NS', 'PIIND.NS', 'PNB.NS', 'POLYCAB.NS', 'POWERGRID.NS', 'PVRINOX.NS', 'RAIN.NS', 'RBLBANK.NS',
#     'RECLTD.NS', 'RELIANCE.NS', 'SAIL.NS', 'SBICARD.NS', 'SBILIFE.NS', 'SBIN.NS', 'SHREECEM.NS', 'SIEMENS.NS', 'SRF.NS',
#     'STAR.NS', 'SUNPHARMA.NS', 'SUNTV.NS', 'SYNGENE.NS', 'TATACHEM.NS', 'TATACOMM.NS', 'TATACONSUM.NS', 'TATAELXSI.NS', 'TATAMOTORS.NS',
#     'TATAPOWER.NS', 'TATASTEEL.NS', 'TCS.NS', 'TECHM.NS', 'THERMAX.NS', 'TIINDIA.NS', 'TORNTPHARM.NS', 'TORNTPOWER.NS', 'TRENT.NS',
#     'TVSMOTOR.NS', 'UBL.NS', 'ULTRACEMCO.NS', 'UNIONBANK.NS', 'UPL.NS', 'VBL.NS', 'VEDL.NS', 'VOLTAS.NS', 'WHIRLPOOL.NS', 'WIPRO.NS',
#     'YESBANK.NS', 'ZEEL.NS', 'ZOMATO.NS'
# ]

nifty50_tickers = [
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

# Sidebar for user input
st.sidebar.title('NIFTY 200 Analysis Dashboard')
selected_ticker = st.sidebar.selectbox("Select a Ticker", nifty50_tickers)

st.sidebar.header("Date Range Selection")
start_date = st.sidebar.date_input("Start Date",datetime(2024, 1, 1))
end_date = st.sidebar.date_input("End Date")
assert(end_date>=start_date)


time_frame = st.sidebar.selectbox("Select Time Frame", ['1d','5d', '1wk', '1mo'])
duration = st.sidebar.selectbox("Select Duration", ['1mo', '3mo', '6mo','1y', '2y','5y'])
analysis_type = st.sidebar.selectbox("Choose Analysis Type", ['Close', 'Heikin-Ashi Close', 'Night', 'Day'])


data = yf.download(selected_ticker, start=start_date, end=end_date, interval=time_frame)

# Fetch stock data from yfinance
# data = yf.download(selected_ticker, period=duration, interval=time_frame)
data.index = pd.to_datetime(data.index)


# Calculating daily return as (Close - Open) * 100 / Open
data['Day Return'] = (data['Close'] - data['Open']) * 100 / data['Open']

# Calculating previous night return as (Open - Previous Close) * 100 / Previous Close
data['Previous Close'] = data['Close'].shift(1)
data['Previous Night Return'] = (data['Open'] - data['Previous Close']) * 100 / data['Previous Close']

# Apply Heikin-Ashi calculation if selected
if analysis_type == 'Heikin-Ashi Close':
    data = heikin_ashi(data)
    daily_close = data['HA_Close']
    daily_returns = daily_close.pct_change() * 100
    daily_returns = daily_returns.dropna()
elif analysis_type == 'Close':
    daily_close = data['Close']
    daily_returns = daily_close.pct_change() * 100
    daily_returns = daily_returns.dropna()
elif analysis_type == 'Night':
    daily_returns = data['Previous Night Return']
    daily_returns = daily_returns.dropna()
elif analysis_type == 'Day':
    daily_returns = data['Day Return']
    daily_returns = daily_returns.dropna()


# Dropping rows with NaN values
data.dropna(inplace=True)

# Calculating mean, variance, skewness, and kurtosis for both returns
mean_day = data['Day Return'].mean()
var_day = data['Day Return'].var()
skew_day = skew(data['Day Return'])
kurt_day = kurtosis(data['Day Return'])+3

mean_previous_night = data['Previous Night Return'].mean()
var_previous_night = data['Previous Night Return'].var()
skew_previous_night = skew(data['Previous Night Return'])
kurt_previous_night = kurtosis(data['Previous Night Return'])+3

# print(daily_returns)

# Sidebar for upper and lower bounds to exclude outliers
st.sidebar.write("### Exclude Outliers from Analysis")
lower_bound = st.sidebar.number_input('Lower Bound (%)', value=-10)
upper_bound = st.sidebar.number_input('Upper Bound (%)', value=10)

# Sidebar for middle %
st.sidebar.write("### Central Percentage")
bound = st.sidebar.number_input('Bound (%)', value=70)
usl = 0.5+(bound/200)
lsl = 0.5-(bound/200)
# kde = stats.gaussian_kde(data)

# # Define the CDF of the KDE
# def kde_cdf(x):
#     return kde.integrate_box_1d(-np.inf, x)

# # Find the value where CDF is 0.7 (i.e., 70% of the area is to the left)
# x_range = np.linspace(min(data), max(data), 1000)
# cdf_values = [kde_cdf(x) for x in x_range]

# Find the x value where the CDF is closest to 0.7


kde = stats.gaussian_kde(daily_returns.values)

def kde_cdf(x):
    return kde.integrate_box_1d(-np.inf, x)

# Find the value where CDF is 0.7 (i.e., 70% of the area is to the left)
x_range = np.linspace(min(daily_returns), max(daily_returns), 1000)
cdf_values = [kde_cdf(x) for x in x_range]

# Find the quantile such that 0.3 of the area is to the right
# quantile_0_8 = stats.norm.ppf(0.9, np.mean(data), np.std(data))
uslx = x_range[np.searchsorted(cdf_values, usl)]
lslx = x_range[np.searchsorted(cdf_values, lsl)]
# print(quantile_0_7)
# Exclude outliers
filtered_returns = daily_returns[(daily_returns >= lower_bound) & (daily_returns <= upper_bound)]

# Calculate mean, median, and variance
mean_return = filtered_returns.mean()
median_return = filtered_returns.median()
variance_return = filtered_returns.var()

# Scatter plot of percent returns at the top of the dashboard (against Date)
st.write(f"## Scatter Plot of Daily Returns for {selected_ticker}")
plt.figure(figsize=(10, 6))
plt.scatter(filtered_returns.index, filtered_returns, color='blue', label='Returns')
plt.axhline(y=mean_return, color='r', linestyle='--', label=f'Mean: {mean_return:.2f}')
plt.axhline(y=median_return, color='g', linestyle='-', label=f'Median: {median_return:.2f}')
plt.title(f'Scatter Plot of Daily Returns for {selected_ticker}')
plt.xlabel('Date')
plt.ylabel('Daily Percent Return')
plt.legend()
st.pyplot(plt)

# Create histogram with mean, median, and variance
st.write(f"### Histogram of Daily Returns for {selected_ticker}")
plt.figure(figsize=(10, 6))
sns.histplot(filtered_returns, kde=True, stat="density", bins=30, color='blue', label='Returns')
plt.axvline(mean_return, color='r', linestyle='--', label=f'Mean: {mean_return:.2f}')
plt.axvline(median_return, color='g', linestyle='-', label=f'Median: {median_return:.2f}')
# Draw the vertical line at the calculated quantile
plt.axvline(uslx, color='black', linestyle=':', label=f'USL = {uslx:.2f}')
plt.axvline(lslx, color='black', linestyle=':', label=f'LSL = {lslx:.2f}')
plt.title(f'Histogram of Daily Returns for {selected_ticker}')
plt.legend(title=f"Variance: {variance_return:.2f}")
st.pyplot(plt)

# Shapiro-Wilk normality test
st.write("### Shapiro-Wilk Test for Normality")
shapiro_test = stats.shapiro(filtered_returns)
st.write(f"**P-value**: {shapiro_test.pvalue:.4f}")
normal = True
if shapiro_test.pvalue > 0.05:
    st.write("Conclusion: The returns are likely normally distributed (Fail to reject H0).")
else:
    st.write("Conclusion: The returns are not normally distributed (Reject H0).")
    normal = False

# If the returns are not normal, apply transformation (Box-Cox)
if not normal:
    st.write("### Applying Box-Cox Transformation to Normalize Returns")
    transformed_returns, lambda_val = boxcox(filtered_returns + abs(filtered_returns.min()) + 1)

    # Scatter plot of transformed returns
    st.write(f"### Scatter Plot of Transformed Returns for {selected_ticker}")
    plt.figure(figsize=(10, 6))
    plt.scatter(filtered_returns.index, transformed_returns, color='purple', label='Transformed Returns')
    plt.axhline(y=np.mean(transformed_returns), color='r', linestyle='--', label=f'Mean: {np.mean(transformed_returns):.2f}')
    plt.axhline(y=np.median(transformed_returns), color='g', linestyle='-', label=f'Median: {np.median(transformed_returns):.2f}')
    plt.title(f'Scatter Plot of Transformed Returns for {selected_ticker}')
    plt.xlabel('Date')
    plt.ylabel('Transformed Daily Return')
    plt.legend()
    st.pyplot(plt)

    # Re-plot histogram for transformed returns
    st.write(f"### Histogram of Transformed Returns for {selected_ticker}")
    plt.figure(figsize=(10, 6))
    sns.histplot(transformed_returns, kde=True, bins=30, color='purple', label='Transformed Returns')
    plt.axvline(np.mean(transformed_returns), color='r', linestyle='--', label=f'Mean: {np.mean(transformed_returns):.2f}')
    plt.axvline(np.median(transformed_returns), color='g', linestyle='-', label=f'Median: {np.median(transformed_returns):.2f}')
    plt.title(f'Histogram of Transformed Returns for {selected_ticker}')
    plt.legend(title=f"Variance: {np.var(transformed_returns):.2f}")
    st.pyplot(plt)

    # Test for normality again after transformation
    st.write("### Shapiro-Wilk Test for Transformed Returns")
    shapiro_test_transformed = stats.shapiro(transformed_returns)
    st.write(f"**P-value**: {shapiro_test_transformed.pvalue:.4f}")
    if shapiro_test_transformed.pvalue > 0.05:
        st.write("Conclusion: Transformed returns are now normally distributed (Fail to reject H0).")
    else:
        st.write("Conclusion: Transformed returns are still not normal (Reject H0).")

    # Update daily_returns to transformed_returns for further analysis
    daily_returns = transformed_returns
else:
    daily_returns = filtered_returns
#     # Scatter plot of transformed returns
#     st.write(f"### Scatter Plot of Returns for {selected_ticker}")
#     plt.figure(figsize=(10, 6))
#     plt.scatter(filtered_returns.index, filtered_returns, color='purple', label='Transformed Returns')
#     plt.axhline(y=np.mean(filtered_returns), color='r', linestyle='--', label=f'Mean: {np.mean(transformed_returns):.2f}')
#     plt.axhline(y=np.median(filtered_returns), color='g', linestyle='-', label=f'Median: {np.median(transformed_returns):.2f}')
#     plt.title(f'Scatter Plot of Transformed Returns for {selected_ticker}')
#     plt.xlabel('Date')
#     plt.ylabel('Transformed Daily Return')
#     plt.legend()
#     st.pyplot(plt)



# ADF Test for stationarity
st.write("### Augmented Dickey-Fuller (ADF) Test for Stationarity")
adf_test = adfuller(daily_returns)
st.write(f"**ADF Statistic**: {adf_test[0]:.4f}")
st.write(f"**P-value**: {adf_test[1]:.4f}")
if adf_test[1] < 0.05:
    st.write("Conclusion: The time series is stationary (Reject H0).")
else:
    st.write("Conclusion: The time series is not stationary (Fail to reject H0).")

# ACF and PACF plots
st.write("### ACF and PACF Plots")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

plot_acf(daily_returns, ax=axes[0])
axes[0].set_title('Autocorrelation (ACF)')

plot_pacf(daily_returns, ax=axes[1])
axes[1].set_title('Partial Autocorrelation (PACF)')

st.pyplot(fig)

# Ljung-Box test on returns before ARMA fitting
st.write("### Ljung-Box Test on Returns (Before ARMA Model Fitting)")
lb_test_returns = acorr_ljungbox(daily_returns, lags=[10], return_df=True)
st.write(f"**P-value**: {lb_test_returns['lb_pvalue'].values[0]:.4f}")
if lb_test_returns['lb_pvalue'].values[0] > 0.05:
    st.write("Conclusion: The residuals do not show autocorrelation (Fail to reject H0).")
else:
    st.write("Conclusion: The residuals show autocorrelation (Reject H0).")

# Sidebar input for ARMA model order (p, d, q)
st.sidebar.write("### ARMA Model Parameters")
p = st.sidebar.number_input("AR value (p)", value=1, min_value=0)
d = st.sidebar.number_input("I value (d)", value=0, min_value=0)
q = st.sidebar.number_input("MA value (q)", value=1, min_value=0)

# Fit ARMA model
st.write(f"### ARMA Model Summary for {selected_ticker} (p={p}, d={d}, q={q})")
arma_model = ARIMA(daily_returns, order=(p, d, q)).fit()
st.write(arma_model.summary())

# Residuals plot for ARMA model
st.write("### Residuals of ARMA Model")
residuals = arma_model.resid
plt.figure(figsize=(10, 6))
plt.scatter(filtered_returns.index, residuals, label='Residuals', color='orange')
plt.axhline(y=residuals.mean(), color='r', linestyle='--', label=f'Mean: {residuals.mean():.2f}')
plt.axhline(y=np.median(residuals), color='g', linestyle='-', label=f'Median: {np.median(residuals):.2f}')
plt.title('Residuals of ARMA Model')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.legend()
st.pyplot(plt)

# R-squared and adjusted R-squared
st.write("### R-squared and Adjusted R-squared")
y_true = daily_returns
y_pred = arma_model.fittedvalues

r_squared = r2_score(y_true, y_pred)
n = len(y_true)
p_model = arma_model.df_model
adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p_model - 1)

st.write(f"**R-squared**: {r_squared:.4f}")
st.write(f"**Adjusted R-squared**: {adj_r_squared:.4f}")

st.write(f"## Day return & Night return histogram of {selected_ticker}")

# Plotting histograms for Daily Return and Previous Night Return
fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Daily Return Histogram
axs[0].hist(data['Day Return'], bins=20, color='blue', edgecolor='black', alpha=0.7)
axs[0].set_title(f'Histogram of Day Return')
axs[0].set_xlabel('Day Return (%)')
axs[0].set_ylabel('Frequency')
axs[0].legend([f"Mean: {mean_day:.2f}\nVariance: {var_day:.2f}\nSkewness: {skew_day:.2f}\nKurtosis: {kurt_day:.2f}"], loc='upper right')

# Previous Night Return Histogram
axs[1].hist(data['Previous Night Return'], bins=20, color='green', edgecolor='black', alpha=0.7)
axs[1].set_title(f'Histogram of Night Return')
axs[1].set_xlabel('Night Return (%)')
axs[1].legend([f"Mean: {mean_previous_night:.2f}\nVariance: {var_previous_night:.2f}\nSkewness: {skew_previous_night:.2f}\nKurtosis: {kurt_previous_night:.2f}"], loc='upper right')

st.pyplot(fig)
