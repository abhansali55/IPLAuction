#installing respective graph theory packages
# !sudo apt-get install python3-dev graphviz libgraphviz-dev pkg-config
# !sudo pip install pygraphviz
# !pip install networkx
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt #to help align the graphs
import networkx as nx #allows us to build the graphs that we need
import pygraphviz as pgv #visualization package for graphs
import pandas as pd #help clean the data
import numpy as np #used for the numeric operations within script
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#since we have newer teams in the dataset, we need to accomondate for this
def modify_data(data_set):
  #taking the mean for each team for the auction type
  grouped = data_set.groupby('Team').mean(numeric_only=True)

  #sorting the values by the teamID so that we can get the symmetric matrix
  #that we originally started with
  grouped.sort_values(by=['teamID'],inplace=True)

  #since these columns were for filtering and sorting purposes, we can drop them
  #now since they aren't too necessary for analysis
  grouped.drop(['year','isMega','teamID'],axis=1,inplace=True)

  #taking the mean values for newer teams and transposing them since they don't match
  #what is in the grouped dataset which then gets combined into a dataframe consisting
  #of the mean values 
  newer = grouped.iloc[8:].T
  grouped.drop(list(grouped.columns[8:]),axis=1,inplace=True)
  combined_data = pd.concat([grouped,newer],axis=1)

  combined_data.reset_index(drop=True,inplace=True)
  teams = pd.DataFrame(list(combined_data.columns),columns=['Team'])

  return pd.concat([teams,combined_data],axis=1)
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#overall network for all of the teams and their respective bidding partners
def build_graph(data_set):
  total_bid_pairs = data_set.iloc[:,1:].values.sum()
  teams = list(data_set.columns[1:])

  G = nx.Graph() #create graph object
  G.add_nodes_from(set(teams)) #verticies for each graph is set of teams

  #taking each team and adding respective bid pairs
  for i in range(len(data_set)):
    fav_team = list(data_set['Team'])[i]
    for j in range(len(teams)):
      edge_weight = np.round(data_set.iloc[i,j+1]/total_bid_pairs,2)
      if (edge_weight > 0) and (fav_team != teams[j]):
        G.add_edge(fav_team, teams[j], weight=edge_weight)
  
  return G
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#building a conditional probabilty graph given that a specific team entered into the auction lot
#can have several types of functions to find all kinds of graphs so long as your data follows the 
#format to build the graph as above 
def graph_given_team(sub_data_set):
  teams = list(sub_data_set.columns[1:])
  subgraphs = []
  for k in range(len(teams)):
    sub_data = sub_data_set[sub_data_set['Team'] == teams[k]]
    subgraphs.append(build_graph(sub_data))
  
  return subgraphs
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#drawing each graph that we have built from the given datasets
def draw_graphs(G, file_path):
  pos = nx.nx_agraph.graphviz_layout(G, prog="neato") #have certain layout for all graphs
  
  nx.draw_networkx_nodes(G, pos, node_size=400) #draw nodes
  nx.draw_networkx_edges(G, pos, alpha=0.5, width=.3) #draw edges

  #labeling nodes and edges for the graph along with getting desired attributes
  nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif")
  nx.draw_networkx_edge_labels(G, pos, nx.get_edge_attributes(G, "weight"), font_size=10)

  #to plot the graph for us to see in code file
  ax = plt.gca()
  ax.margins(0.15) #change size of page for the graph
  plt.tight_layout()
  plt.savefig(file_path + '.png', format="PNG") #saving each graph to your respected folder
  plt.show() #shows the figure in the output
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
path_file = '/content/drive/MyDrive/IPLAuction'

big_data = pd.read_csv(path_file + '/bidpairyearly.csv')

#saving the images into these folders for your respective computer
folders = ['/Mega','/Mini']

#this loop was created to separate the mega and mini auctions
for j in range(2):
  auction_type = modify_data(big_data[big_data['isMega'] == j])
  graphs_to_draw = graph_given_team(auction_type) #returns list of the subgraphs to draw

  #this graph is the probaiblity that any two teams will be bidding pairs in any lot
  #adding it into the list of graphs to draw out
  big_network = pd.DataFrame(np.tril(auction_type.values),
                                       columns=list(auction_type.columns))
  graphs_to_draw.insert(0,build_graph(big_network))

  #loops through the list and draws out each graph in the list  
  for i in range(len(graphs_to_draw)):
    full_path = path_file + folders[j] + '/' + auction_type.columns[i] #where to save each graph drawn
    draw_graphs(graphs_to_draw[i], full_path)
