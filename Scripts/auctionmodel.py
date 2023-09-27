import numpy as np #specific operations completed with this library
import pandas as pd #helps prepare the data
from sklearn.ensemble import RandomForestRegressor #used for posiitons and teams bidding
from sklearn.linear_model import LinearRegression #used for bids and price of player
from sklearn.linear_model import LogisticRegression #used for bids and price of player
from sklearn.model_selection import train_test_split #helps split data for models
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#combine the teams that were in the auction along with the total bids and price
#for all players given the year and position on the field
def prepare_data(file_path):
  #reading in this file so that we can gather all teams that bid on player
  player_sequence = pd.read_csv(file_path + '/playerbidsequence.csv')

  #main columns to help us group the data from player_sequence dataframe
  useful_cols = ['playerID','playerName', 'year','playerRole']

  #getting all of the distinct teams that bid on the player in the auction
  distinct_teams = pd.DataFrame(player_sequence.groupby(useful_cols)['teamName'].unique()).explode('teamName')

  #since all of the teams are categorical variables, we will need to use 1s and 0s to classify
  #the value will be 1 if a team bid on the player and 0 if they did not in each lot
  num_team = pd.DataFrame(distinct_teams.pivot_table(index=useful_cols, columns='teamName', 
                                                        aggfunc=lambda x: 1, fill_value=0))
  
  #this is aggregated auction data for each player with the final price of player
  player_auction = pd.read_csv(file_path + '/playerauction.csv')

  return pd.concat([player_auction,pd.DataFrame(num_team.values, columns=num_team.columns)],axis=1)
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#the purposes of this is to take all of the auctions from the previous years which will give us 
#an example of what an auction can look like. to account for randomness within each trial,
#it makes more sense to run hundreds of trials and get many different results
def models(data):
  N = 1000
  different_auctions = []

  i = 0
  while i < N:
    bids = data['numBids'] #number of bids for a lot
    price = data['price'] #final price of the lot
    auction_type = data['isMega'] #determines mega or mini auction for the lot

    #taking the categorical data of the positions and converting them into 1s and 0s
    #as dummy variables so that we can use them for the model of choice
    role_dummies = pd.get_dummies(data['playerRole'])
    
    #these are the teams that bid on each lot
    teams = data.iloc[:,7:]

    #combine all the data together
    X = pd.concat([role_dummies,auction_type, bids, price, teams], axis=1)

    #split into training and testing sets. industry practice for test size is anywhere from 20-30% so we will go with .2 as test size here
    #when we are using the training and testing data, we will need to section off the column which we are trying to predict  
    X_train, X_test, y_bids_train, y_bids_test, y_roles_train, y_roles_test, y_prices_train, y_prices_test, y_teams_train, y_teams_test, y_classify_auction_train, y_classify_auction_test = train_test_split(
    X.values, bids.values, role_dummies.values, price.values, teams.values, auction_type.values, test_size=.2)

    #since there are several columns we want to predict, it is easier to have all the values for the training and testing
    #with this in mind, it makes it easier for us to predict multiple columns by filtering out the column(s) as needed  
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)

    #since there are several different classifiers in the dataset, using Random Forest to generate sample
    #player posiitons and teams that bid in each lot would be much more effective compared to Logistic Regression
    #for both the player position and the teams as well
    role_model = RandomForestRegressor()
    role_model.fit(X_train.iloc[:,np.r_[4:len(X_train.columns)]], y_roles_train)

    teams_model = RandomForestRegressor()
    teams_model.fit(X_train.iloc[:,np.r_[7:len(X_train.columns)]], y_teams_train)

    #since the price is continuous, can simply use linear regression in which the dummy
    #variables are categorical variables within the model itself
    price_model = LinearRegression()
    price_model.fit(X_train.iloc[:,np.r_[:6,7:len(X_train.columns)]], y_prices_train)

    #even though bid numbers must be integers, using the linear model to predict the number
    #of bids for a lot is still viable  
    bids_model = LinearRegression()
    bids_model.fit(X_train.iloc[:,np.r_[:5,6:len(X_train.columns)]], y_bids_train)

    #the original model for predicting classifiers would be logistic regression
    #which is what we would be doing here since we are essentially predicting here 
    #based on the data whether the lot would be within a mega or mini auction
    classify_auction_model = LogisticRegression(solver='lbfgs', max_iter=1000)
    classify_auction_model.fit(X_train.iloc[:,np.r_[:4,5:len(X_test.columns)]], y_classify_auction_train)

    #making predictions for sample auctions for each of the player role, bids, price, 
    #and teams that bid on a player in a lot during the auction
    y_roles_pred = role_model.predict(X_test.iloc[:,np.r_[4:len(X_test.columns)]])
    y_teams_pred = teams_model.predict(X_test.iloc[:,np.r_[7:len(X_test.columns)]])
    y_prices_pred = price_model.predict(X_test.iloc[:,np.r_[:6,7:len(X_test.columns)]])
    y_bids_pred = bids_model.predict(X_test.iloc[:,np.r_[:5,6:len(X_test.columns)]])
    y_auction_type_pred = classify_auction_model.predict(X_test.iloc[:,np.r_[:4,5:len(X_test.columns)]])

    #taking predictions and converting them into dataframes
    predicted_roles = pd.DataFrame(y_roles_pred, columns=role_dummies.columns).idxmax(axis=1)
    predicted_bids = pd.DataFrame(np.round(y_bids_pred)).astype(int)
    predicted_prices = pd.DataFrame(y_prices_pred).round(decimals=2)
    predicted_auction_type = pd.DataFrame(np.round(y_auction_type_pred)).astype(int)
    predicted_teams = pd.DataFrame(np.round(y_teams_pred),
                                   columns=data.iloc[:,7:].columns)
    
    #need to concat these first since they don't have column names at first
    auction = pd.concat([predicted_roles,predicted_bids,
                         predicted_prices,predicted_auction_type],axis=1)

    #naming the columns for each auction since they aren't named due to the dummy
    #variable that was added for the player positions
    auction.columns = ['role', 'new_bids', 'new_prices', 'new_auction_type']

    auction = pd.concat([auction,predicted_teams],axis=1)

    different_auctions.append(auction)
    i+=1
  
  return different_auctions
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#running the model and then writing it to csv files which will consist of
#one sample mega auction and one sample mini auction from the datasets
def write_to_csv():
  path = '/content/drive/MyDrive/IPLAuction'

  different_auctions = pd.concat(models(prepare_data(path)),axis=0)

  #for the pattern that was detected, it makes sense for us to only take lot with at least four bids
  #this can be adjusted accordingly if we need more data from player auctions
  usable_bids = different_auctions[different_auctions['new_bids'] >= 4]

  #when there is more than four bids, it is impossible to have less than 
  #two teams in the lot so need to throw away those lots from this dataset
  usable_bids = usable_bids[usable_bids.iloc[:,4:].sum(axis=1) >= 2]

  #not possible to have a price of a lot be less than or equal to zero
  usable_bids = usable_bids[usable_bids['new_prices'] > 0]

  mega_auction = usable_bids[usable_bids['new_auction_type'] == 1]
  mini_auction = usable_bids[usable_bids['new_auction_type'] == 0]  

  #specific way to order the auctions
  order_cols = ['new_bids','new_prices']
  
  #returning an example of what a mega and mini auction will look like
  sample_mega = mega_auction.sample(n=100).sort_values(by=order_cols,ascending=False)
  sample_mini = mini_auction.sample(n=25).sort_values(by=order_cols,ascending=False)

  #saving the data into csv files
  sample_mega.drop(['new_auction_type'],axis=1).to_csv(path + '/megaauction.csv',index=False)
  sample_mini.drop(['new_auction_type'],axis=1).to_csv(path + '/miniauction.csv',index=False)

  #teams that entered into the lot
  big = mega_auction.iloc[:,4:]
  small = mini_auction.iloc[:,4:]
  
  #generating percent of lots that each team enters the auction given that it is
  #either a mega or mini auction
  lot_avg = pd.concat([pd.DataFrame(np.round(small.sum(axis=0)/small.values.sum(),4),columns=['mini']),
                       pd.DataFrame(np.round(big.sum(axis=0)/big.values.sum(),4),columns=['mega'])],
                      axis=1)

  lot_avg.sort_values(by=list(lot_avg.columns),ascending=False,inplace=True).to_csv(path + '/avgteambid.csv',index=False)

  return 'Model complete.'
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
write_to_csv() #function to run
