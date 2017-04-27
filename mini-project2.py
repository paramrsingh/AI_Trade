# Param Singh
# Mini-Project2 - AI Trade
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.data import morningstar
from sklearn.ensemble import RandomForestClassifier
from collections import deque
import numpy as np

def initialize(context):
    
    # Create and attach an empty Pipeline.
    my_pipe = make_pipeline()
    attach_pipeline(my_pipe, 'my_pipeline')
    # Classifier
    context.classifier = RandomForestClassifier()
    # Variable Handlers
    context.window_length = 10
    context.recent_prices = deque(maxlen=context.window_length + 2)
    context.recent_value = deque(maxlen=context.window_length + 2)
    context.X = deque(maxlen=200)
    context.Y = deque(maxlen=200)
    context.prediction = 0
    
    # Algo
    schedule_function(aitrade, date_rule=date_rules.every_day(), time_rule=time_rules.market_open(hours=0, minutes=1))                     
    
    # Slippage and commission, assuming i'm getting a deal
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1))
    set_commission(commission.PerTrade(cost=.75))

    
def make_pipeline(): 
    # Latest p/e ratio.
    pe_ratio = morningstar.valuation_ratios.pe_ratio.latest    
    # Top open securities filter (high dollar volume securities)
    high_dollar_val = USEquityPricing.open.latest.top(10)    
    # Top 5 stocks ranked by p/e ratio
    top_pe_stocks = pe_ratio.top(5, mask=high_dollar_val)
    
    return Pipeline(columns={
                'pe_ratio': pe_ratio,
              },screen = top_pe_stocks)
    
    
def before_trading_start(context, data):
    daily_contenders = pipeline_output('my_pipeline')
    #assume that p/e ratios less than 20 are a 'buy' indicator
    context.pebuys = daily_contenders.loc[daily_contenders['pe_ratio'] < 20]
    
def aitrade(context, data):  
    #get price history  
    for security in context.pebuys.index.tolist():
        
        context.recent_prices.append(data.history(security, 'price', 200, '1m'))
        context.recent_value.append([data.history(security, 'price', 200, '1m')])
        
        changes = np.diff(context.recent_prices) > 0
        values = np.array(context.recent_value).flatten()
        context.X.append(values[:-1])
        context.Y.append(changes[-1])
            
        context.classifier.fit(context.X, context.Y)
        predictor = data.history(security, 'price', 200, '1d')

        # Getting an array of predictions, currently setup as a single tree with multiple features
        context.prediction = context.classifier.predict(predictor[:-1])
        # get median prediction from result array, >= .5 means there is generally an upward trend in price
        # not the most sophisticated but attempting to use pipeline with the classifier
        predict_ave = np.average(context.prediction)
        if predict_ave < 0.5:
            # call order management
            buy_mngmt(security)
        else:
            sell_mngmt(security)
        
        #reset for the next security
        context.recent_prices.clear()
        context.recent_value.clear()
        context.X.clear()
        context.Y.clear()
        context.prediction = 0
        predict_ave = 0

        
def buy_mngmt(sec_buy):
    order_percent(sec_buy, .1)
    
def sell_mngmt(sec_sell):
    order_percent(sec_sell, -.025)