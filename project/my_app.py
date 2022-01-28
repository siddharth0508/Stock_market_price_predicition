import streamlit as st
import pandas as pd
from pandas_datareader import data
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pandas import datetime
from statsmodels.tsa.arima.model import ARIMA
import plotly.express as px
import yfinance as yf
import pmdarima as pm

st.set_page_config(page_title='Stock Prediction',page_icon="📈",layout='wide',initial_sidebar_state='collapsed')

st.markdown(
    """
<style>
.main{
    background-image: linear-gradient(#304352,#d7d2cc); 
    color:White;
    text-align:left;
}
.sidebar .sidebar-content { 
    background-image: linear-gradient(#8e9eab,#eef2f3); 
    color:White; 
}
</style>
""",
    unsafe_allow_html=True,
)
st.sidebar.header('ENTER STOCK NAME')
input=st.sidebar.text_input('\n')

header=st.container()
data_set=st.container()

with header:
    st.title('STOCK Price Prediction')
    col1, col2 = st.columns(2)
    col1.image(yf.Ticker(input).info['logo_url'])
    col2.header(yf.Ticker(input).info['longName'])
    st.write(yf.Ticker(input).info['longBusinessSummary'])

to_date=datetime.now()
start_date = to_date - relativedelta(years=5)
dates = pd.date_range(start_date, to_date)
df = data.DataReader(input, 'yahoo', dates[0],dates[-1])
pd.DataFrame(data=df, columns=['Adj Close'], index=df.index.values)
df['Date']=df.index
df.reset_index(inplace=True, drop=True)

with data_set:
    st.header('Data Set')
    st.dataframe(df)
    fig = px.line(df, x='Date', y='Close')
    fig.update_xaxes(rangeslider_visible=True)
    st.header('Data Set Graph')
    st.plotly_chart(fig)

    train_data, test_data = df[0:int(len(df) * 0.7)], df[int(len(df) * 0.7):]
    training_data = train_data['Close'].values
    test_data = test_data['Close'].values
    history = [x for x in training_data]
    model_predictions = []
    next_n_days = []
    N_test_observations = len(test_data)

    automodel = pm.auto_arima(training_data, start_p=1, d=None, start_q=1, max_p=5, max_d=2, max_q=5, seasonal=False,
                              trace=True)

    for time_point in range(N_test_observations):
        model = ARIMA(history, order=automodel.order)
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        model_predictions.append(yhat)
        true_test_value = test_data[time_point]
        history.append(true_test_value)

    model = ARIMA(history, order=automodel.order)
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    next_n_days.append(yhat)

    test_set_range = df[int(len(df) * 0.7):].index

    gr={'Actual':test_data,'Predicted':model_predictions}
    gr1=pd.DataFrame(gr)
    figure2 = px.line(gr1,template='simple_white')
    figure2.update_layout(xaxis_title="index",yaxis_title="Stock Price")
    figure2.update_xaxes(rangeslider_visible=True)
    st.header('Actual VS Predicted')
    st.plotly_chart(figure2)
    st.header('Next DAY Prediction')
    r=df.tail(1).Close
    p=float(r.values[0])
    if p<yhat:
        st.subheader(str(yhat)+"▲")
    else:
        st.subheader(str(yhat)+"▼")

    st.balloons()