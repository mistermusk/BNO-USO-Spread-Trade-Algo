'''

BNO/USO Pairs Trading Strategy
------------------------------

--> Description of strategy: this strategy seeks to find positive returns 
in trading a strategy that takes advantage of the deviations in the spread
between BNO (United States Brent Oil Fund) and USO (United States Oil Fund),
oil price index funds that trade at a 0.972671 correlation to each other. The 
idea behind the strategy is that the spread between the two index funds should,
overtime, revert back to the mean and that we can find returns in riding the moves
back toward the mean. 

The spread is suspected to widen or shrink based off of the reflective natures of
the actual assets themselves. BNO tracks the daily price change in Brent Crude, which 
is widely considered the global oil price index, while USO tracks the daily price
change in West Texas Intermediate Crude. While both track the overall price of sweet,
light crude, Brent generally trades at a premium to WTI becuase of a number of global
macroeconomic factors (ex: transport costs, glutten supply in the Americas, etc.), and
is also more affected by non-U.S. events such as conflicts in the Middle East, E.U.
crisis events, and other Euro-Asia-Africa factors. 

Knowing that the two assets (BNO and USO) trade in almost perfect correlation - but that
there are macro factors that cause the spread to change - we treat those macro variables 
as mostly noise that can cause spread differentials in the short-term but over the long-
term the spread reverts back to the mean. 

To take advantage of the deviations in the mean we decided to incorporate a fairly tight
trade zone that is relatively selective on opportunities in which to trade and to try to
be as risk-off (holding 100% cash) as much as possible. To establish a signal in which to trade
we wanted the spread to be more than just outside of the first standard deviation, but we
also found that establishing a signal past the second deviation didn't give us enough
opportunities to take advantage of and basically left returns at zero. Knowing that we 
wanted the trade signal to initiate past the first standard deviation but before the second
standard deviation we played with different variations and found that initiating the trade at
1.8 gave us the best chance at reproduceable alpha and positive Sharpe ratios. 

-------------------------------------------------------------------------------------------------

--> Testing notes: backtested with different capital allocations of $10 million, $1 million, $100
thousand, and $10 thousand. 

Performace generally increased with allocation size. Possible theory of why this occured is due
to the overall illiquidity of BNO and that, with larger size of the allocation, orders of long 
and short positions on BNO would go unfilled at the end of the day skewing the allocation more 
to being biased towards USO. More work needs to be done to actually incorporate this bias since
it ended up increasing returns.

'''

import numpy as np

def initialize(context):
    
    # setting the benchmark to XOP (SPDR's U.S. Oil Producer & Explorer ETF) in order to 
    # compare returns to fluxuations in oil price without using either one of the strategy's
    # underlying assets (BNO or USO) as the benchmark. To toggle between setting the benchmark
    # from XOP to the default benchmark of the SPY, just block comment the line below
    set_benchmark(sid(32279))
    
    # scheduling the check_spread function to run once a day, every trading day after one hour 
    # in to the trading day to (in theory) reduce the uneeded noise at the very beggining of
    # the trading session
    schedule_function(check_spread, date_rules.every_day(), time_rules.market_open(minutes=60))
    
    # USO (United States Oil Fund)
    context.uso = sid(28320)
    
    # BNO (United States Brent Oil Fund)
    context.bno = sid(39699)
    
    # Used to check if we are already shorting one of the assets to make sure we 
    # don't initialize a short when we already have a short on that asset since
    # we don't want to rebalance already open positions, just close them
    context.short_bno = False
    context.short_uso = False

# this function checks the rolling 50-day mean spread (normalized by using a z-score) 
# to help us determine if we want to initialize a trade or not. If the z-score reaches 
# any of our points that we determined then a simple logic tree determines what trade
# to make based off of the criteria
def check_spread(context, data):
    
    bno = context.bno
    uso = context.uso
    
    # Returns a 50-day history of asset prices
    prices = data.history([bno,uso], 'price', 50, '1d')
    
    # Returns the current price of the assets 
    current_prices = prices.iloc[-1:]
    
    # Calculates the rolling 50-day mean of the spread
    mavg = np.mean(prices[bno] - prices[uso])
    
    # Calculates the rolling 50-day standard deviaiton of the spread
    std = np.std(prices[bno] - prices[uso])
    
    # Calculates the current spread of the prices
    spread = np.mean(current_prices[bno] - current_prices[uso])
    
    # First logic step helps block zero division errors
    if std > 0:
        
        # Calculates the z-score of the spread
        zscore = (spread - mavg) / std
        
        # If the z-score of the spread is above 1.8, menaing the BNO's premium
        # to USO is higher than normal, we decide to short BNO and long USO since
        # we expect the spread to revert to the mean, meaning that BNO's price will
        # decrease relative to USO and USO's price will increase relative to BNO
        if zscore > 1.8 and not context.short_bno:
            order_target_percent(bno, -0.5)
            order_target_percent(uso, 0.5)
            
            # Tells us that we are now short BNO and long USO
            context.short_bno = True
            context.short_uso = False
        
        # If the z-score of the spread is less than -1.8, meaning that USO's discount
        # to BNO is less than normal, we decide to short USO and long BNO since we expect 
        # the spread to revert back to the mean, meaning that USO's price will decrease
        # relative to BNO and BNO's price will increase relative to USO 
        elif zscore < -1.8 and not context.short_uso:
            order_target_percent(bno, 0.5)
            order_target_percent(uso, -0.5)
            
            # Tells us that we are now long BNO and short USO
            context.short_bno = False
            context.short_uso = True
        
        # If the absolute value of the z-score falls back below 1.0 then we will exit the trade 
        # and hopefully take our profits
        elif abs(zscore) < 1.0:
            order_target_percent(bno, 0.0)
            order_target_percent(uso, 0.0)
            
            # Tells us that we are not shorting either asset
            context.short_bno = False
            context.short_uso = False
            
        # Records the 50-day rolling Z-score    
        record(zscore = zscore)