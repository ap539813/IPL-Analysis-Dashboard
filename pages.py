import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.metrics.pairwise import cosine_similarity


def recommend_players(player_name, player_type = None, n=5, player_features_scaled = None):
    """
    Recommends 'n' players similar to the given player based on the similarity matrix.
    
    Parameters:
    - player_name (str): Name of the player.
    - player_type (str): Type of the player (Batsman, Bowler, All-Rounder).
    - n (int): Number of similar players to recommend.
    - player_features_scaled: Final features of the players
    
    Returns:
    - List of recommended players.
    """

    # Create Similrity Matrix
    similarity_matrix = cosine_similarity(player_features_scaled)
    similarity_df = pd.DataFrame(similarity_matrix, index=player_features_scaled.index, columns=player_features_scaled.index)

    
    # Filter based on player type if given
    if player_type:
        possible_recommendations = player_features_scaled[player_features_scaled[f'TYPE_{player_type}'] == 1].index
    else:
        possible_recommendations = player_features_scaled.index
        
    # Get similar players
    similar_players = similarity_df[player_name].loc[possible_recommendations]

    if len(similar_players.shape) > 1:
        similar_players = similar_players.iloc[:, 0]

    similar_players = similar_players.sort_values(ascending=False)
    
    print(similar_players.index)
    # Remove the player himself/herself from the recommendations and get top n players
    recommended_players = similar_players.iloc[1:n+1]
    
    return recommended_players



def main(st, df_balls, df_combined, players, player_insight_df_batsman, player_insight_df_bowler, df_ipl, player_df):
    # Counting matches for both batsmen and bowlers
    df1 = df_balls[['id', 'batsman']]
    df1.columns = ['id', 'PLAYER']
    df2 = df_balls[['id', 'bowler']]
    df2.columns = ['id', 'PLAYER']

    # Combining the counts for players
    matches_played = pd.concat([df1, df2]).value_counts().reset_index()['PLAYER'].value_counts().reset_index()
    matches_played.columns = ['PLAYER', 'MATCHES_PLAYED']



    # Number of matches won by each player (based on player's team)
    player_wins = {}


    for player in players:
        batsman_wins = df_combined[(df_combined['batsman'] == player) & (df_combined['winner'] == df_combined['batting_team'])]
        bowler_wins = df_combined[(df_combined['bowler'] == player) & (df_combined['winner'] == df_combined['bowling_team'])]
        
        # Consider unique matches only
        unique_batsman_wins = batsman_wins.drop_duplicates(subset=['id'])
        unique_bowler_wins = bowler_wins.drop_duplicates(subset=['id'])
        
        total_wins = len(unique_batsman_wins) + len(unique_bowler_wins)
        player_wins[player] = total_wins

    matches_won = pd.DataFrame(player_wins, index = ['COUNT']).T.reset_index()
    matches_won.columns = ['PLAYER', 'MATCHES_WON']


    # Merging the features
    player_features = matches_played.merge(matches_won, on='PLAYER', how='outer')
    print(player_features.shape)
    # player_features = player_features.merge(wickets_taken, on='PLAYER', how='outer')
    # player_features = player_features.merge(matches_won, on='PLAYER', how='outer')
    player_features = player_features.merge(player_insight_df_batsman, on='PLAYER', how='outer')
    # print(player_features.shape)
    player_features = player_features.merge(player_insight_df_bowler, on='PLAYER', how='outer')
    # print(player_features.shape)
    player_features = player_features.merge(df_ipl[['PLAYER', 'COST IN ₹ (CR.)', 'TYPE_ALL-ROUNDER', 'TYPE_BATTER', 'TYPE_BOWLER', 'TYPE_WICKETKEEPER']], on='PLAYER', how='left')
    # print(player_features.shape)
    player_features = player_features.merge(player_df[['PLAYER', 'Batting_Hand', 'Bowling_Skill', 'Country', 'Is_Umpire', 'AGE']], on='PLAYER', how='left')
    # print(player_features.shape)

    player_features.fillna(0, inplace=True)

    player_features.set_index('PLAYER', inplace=True)



    # Scaling the features
    player_features_scaled = player_features.copy()[[col for col in player_features.columns if col not in ['COST IN ₹ (CR.)', 'Batting_Hand', 'Bowling_Skill', 'Country']]]
    player_features_scaled = player_features_scaled.astype(float)

    scaler = StandardScaler()
    player_features_scaled.iloc[:, :-5] = scaler.fit_transform(player_features_scaled.iloc[:, :-5])

    st.session_state['player_features'] = player_features
    st.session_state['player_features_scaled'] = player_features_scaled
    st.session_state['player_df'] = player_df
    st.session_state['df_combined'] = df_combined

    # st.dataframe(player_features)
    return st

