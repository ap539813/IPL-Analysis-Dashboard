import os
import json
import altair as alt
import streamlit as st

import matplotlib.pyplot as plt
import networkx as nx

from add_style import local_css
from pages import main, recommend_players

import pandas as pd
import numpy as np



css_file_path = 'Style/style.css'


## Basic setup and app layout
# Set the Streamlit app configuration
st.set_page_config(page_title="Cricket Analysis", layout="wide")

alt.renderers.set_embed_options(scaleFactor=2)

local_css(css_file_path)




if 'preset_done' not in st.session_state:
    # Load the dataset into a DataFrame
    df_balls = pd.read_csv('IPL Data/IPL Ball-by-Ball 2008-2020.csv')

    # Load the second dataset into a DataFrame
    df_matches = pd.read_csv('IPL Data/IPL Matches 2008-2020.csv')

    # Filter matches data for years 2015 to 2020
    df_matches = df_matches[(df_matches['date'] >= '2015-01-01') & (df_matches['date'] <= '2020-12-31')]

    # Filter ball-by-ball data for the filtered matches
    df_balls = df_balls[df_balls['id'].isin(df_matches['id'])]

    df_ipl = pd.read_csv('IPL Data/ipl_2022_dataset.csv', index_col = 0)
    df_ipl.columns = [col.upper() for col in df_ipl.columns]

    df_names_map = pd.read_csv('IPL Data/data_merge.csv', index_col = 0)

    
    name_map_dict = {(df_names_map['First Name'][i] + ' ' + df_names_map['LastName_x'][i]):df_names_map['Data1'][i] for i in df_names_map.index if type(df_names_map['Data1'][i]) == str}
    df_ipl['PLAYER'] = df_ipl['PLAYER'].replace(name_map_dict)

    df_ipl = pd.get_dummies(df_ipl, columns=['TYPE'])
    df_ipl.drop_duplicates(inplace = True)
    df_ipl = df_ipl.groupby('PLAYER').max().reset_index()

    player_df = pd.read_csv('IPL Data/Player.csv')

    player_df.DOB = pd.to_datetime(player_df.DOB, format = '%d-%b-%y')

    player_df['AGE'] = (pd.to_datetime('today') - player_df.DOB).astype('<m8[Y]')

    df_combined = df_balls.join(df_matches.set_index('id'), on = 'id', how = 'inner', rsuffix = '_ball')

    player_df['PLAYER'] = player_df['PLAYER'].replace(name_map_dict)
    player_df['PLAYER'] = player_df['PLAYER'].replace(name_map_dict)
    player_df = player_df.groupby('PLAYER').max().reset_index()


    df_balls['batsman'] = df_balls['batsman'].replace(name_map_dict)
    df_balls['bowler'] = df_balls['bowler'].replace(name_map_dict)
    df_combined['batsman'] = df_combined['batsman'].replace(name_map_dict)
    df_combined['bowler'] = df_combined['bowler'].replace(name_map_dict)

    players = pd.concat([df_combined['batsman'], df_combined['bowler']]).unique()

    # Grouping by batsman to calculate the required metrics
    batsman_group = df_balls.groupby('batsman')

    # 1. RUNS_SCORED
    runs_scored = batsman_group['batsman_runs'].sum()

    # 4. NUM_SIXES
    num_sixes = batsman_group.apply(lambda x: (x['batsman_runs'] == 6).sum())

    # 5. NUM_FOURS
    num_fours = batsman_group.apply(lambda x: (x['batsman_runs'] == 4).sum())

    # Total balls faced by each batsman
    balls_faced = batsman_group.size()

    # 3. BATTING_STRIKE_RATE
    batting_strike_rate = (runs_scored / balls_faced) * 100

    # Number of times batsman got out
    num_outs = batsman_group['is_wicket'].sum()

    # 2. BATTING_AVERAGE
    batting_average = runs_scored / num_outs
    batting_average[num_outs == 0] = runs_scored[num_outs == 0]

    # 6. BRPI (Batting Runs Per Innings)
    total_innings = df_balls.drop_duplicates(subset=['id', 'batsman'])['batsman'].value_counts()
    brpi = runs_scored / total_innings

    # 7. NOT_OUT
    last_match_id = df_balls.groupby('batsman')['id'].max()
    last_out = df_balls.groupby(['id', 'batsman'])['is_wicket'].max().reset_index()
    # Correcting the NOT_OUT calculation
    last_out = last_out.drop_duplicates(subset='batsman', keep='last').set_index('batsman')
    not_out = last_out['is_wicket'] == 0

    # Create player_insight_df_batsman dataframe again
    player_insight_df_batsman = pd.DataFrame({
        'RUNS_SCORED': runs_scored,
        'BATTING_AVERAGE': batting_average,
        'BATTING_STRIKE_RATE': batting_strike_rate,
        'NUM_SIXES': num_sixes,
        'NUM_FOURS': num_fours,
        'BRPI': brpi,
        'NOT_OUT': not_out
    })


    # Grouping by bowler to calculate the required metrics
    bowler_group = df_balls.groupby('bowler')

    # 1. WICKETS_TAKEN
    wickets_taken = bowler_group['is_wicket'].sum()

    # Total balls bowled by each bowler
    balls_bowled = bowler_group.size()

    # Total runs given by each bowler
    runs_given = bowler_group['total_runs'].sum()

    # 2. AVG_RUN_PER_BALL
    avg_run_per_ball = runs_given / balls_bowled

    # 3. NUM_SIXES_GIVEN
    num_sixes_given = bowler_group.apply(lambda x: (x['batsman_runs'] == 6).sum())

    # 4. NUM_FOURS_GIVEN
    num_fours_given = bowler_group.apply(lambda x: (x['batsman_runs'] == 4).sum())

    # Create player_insight_df_bowler dataframe
    player_insight_df_bowler = pd.DataFrame({
        'WICKETS_TAKEN': wickets_taken,
        'AVG_RUN_PER_BALL': avg_run_per_ball,
        'NUM_SIXES_GIVEN': num_sixes_given,
        'NUM_FOURS_GIVEN': num_fours_given
    })




    # Extracting year from the date and merging it with the df_balls dataframe
    df_balls = df_balls.merge(df_matches[['id', 'date']], on='id')
    df_balls['year'] = pd.to_datetime(df_balls['date']).dt.year

    # Batsman metrics year-wise
    # BATTING_AVERAGE
    runs_per_year = df_balls.groupby(['batsman', 'year'])['batsman_runs'].sum()
    outs_per_year = df_balls.groupby(['batsman', 'year'])['is_wicket'].sum() + 0.00000001
    batting_average_yearwise = runs_per_year / outs_per_year

    # BATTING_STRIKE_RATE
    balls_faced_per_year = df_balls.groupby(['batsman', 'year']).size()
    batting_strike_rate_yearwise = (runs_per_year / balls_faced_per_year) * 100

    # NOT_OUT
    last_match_id_per_year = df_balls.groupby(['batsman', 'year'])['id'].max()
    last_out_per_year = df_balls.groupby(['id', 'batsman', 'year'])['is_wicket'].max().reset_index()
    not_out_yearwise = last_out_per_year[last_out_per_year['id'].isin(last_match_id_per_year)].groupby(['batsman', 'year'])['is_wicket'].max() == 0

    # Reshape the dataframes to have year as columns
    batting_average_yearwise = batting_average_yearwise.unstack(level=-1)
    batting_strike_rate_yearwise = batting_strike_rate_yearwise.unstack(level=-1)
    not_out_yearwise = not_out_yearwise.unstack(level=-1)

    # Update player_insight_df_batsman dataframe
    player_insight_df_batsman = pd.concat([
        player_insight_df_batsman, 
        batting_average_yearwise.add_prefix('BATTING_AVERAGE_'), 
        batting_strike_rate_yearwise.add_prefix('BATTING_STRIKE_RATE_'),
        not_out_yearwise.add_prefix('NOT_OUT_')
    ], axis=1)

    # Bowler metrics year-wise
    # WICKETS_TAKEN
    wickets_per_year = df_balls.groupby(['bowler', 'year'])['is_wicket'].sum()

    # AVG_RUN_PER_BALL
    runs_given_per_year = df_balls.groupby(['bowler', 'year'])['total_runs'].sum()
    balls_bowled_per_year = df_balls.groupby(['bowler', 'year']).size()
    avg_run_per_ball_yearwise = runs_given_per_year / balls_bowled_per_year

    # NUM_SIXES_GIVEN
    num_sixes_given_yearwise = df_balls[df_balls['batsman_runs'] == 6].groupby(['bowler', 'year']).size()

    # NUM_FOURS_GIVEN
    num_fours_given_yearwise = df_balls[df_balls['batsman_runs'] == 4].groupby(['bowler', 'year']).size()

    # Reshape the dataframes to have year as columns
    wickets_per_year = wickets_per_year.unstack(level=-1)
    avg_run_per_ball_yearwise = avg_run_per_ball_yearwise.unstack(level=-1)
    num_sixes_given_yearwise = num_sixes_given_yearwise.unstack(level=-1)
    num_fours_given_yearwise = num_fours_given_yearwise.unstack(level=-1)

    # Update player_insight_df_bowler dataframe
    player_insight_df_bowler = pd.concat([
        player_insight_df_bowler, 
        wickets_per_year.add_prefix('WICKETS_TAKEN_'),
        avg_run_per_ball_yearwise.add_prefix('AVG_RUN_PER_BALL_'),
        num_sixes_given_yearwise.add_prefix('NUM_SIXES_GIVEN_'),
        num_fours_given_yearwise.add_prefix('NUM_FOURS_GIVEN_')
    ], axis=1)

    player_insight_df_batsman.reset_index(inplace = True)
    player_insight_df_bowler.reset_index(inplace = True)

    player_insight_df_batsman.columns = ['PLAYER'] + list(player_insight_df_batsman.columns)[1:]
    player_insight_df_bowler.columns = ['PLAYER'] + list(player_insight_df_bowler.columns)[1:]

    # st.session_state['preset_done'] = True
    



if __name__ == '__main__':
    st.sidebar.markdown('## Select Analysis Type')
    analysis_type = st.sidebar.radio('Type', ['Build Your Team', 'Network Analysis', 'Team Statistics', 'Player Value'])
    if ('player_features' not in st.session_state) or ('player_features_scaled' not in st.session_state):
        main(st, df_balls, df_combined, players, player_insight_df_batsman, player_insight_df_bowler, df_ipl, player_df)


    if analysis_type == 'Build Your Team':
        # Recommendation Interface
        st.title("Select Your Cricket Team")

        # Fetch players based on type
        batsmen = st.session_state['player_features'][st.session_state['player_features']['TYPE_BATTER'] == 1].reset_index()
        bowlers = st.session_state['player_features'][st.session_state['player_features']['TYPE_BOWLER'] == 1].reset_index()
        wicket_keepers = st.session_state['player_features'][st.session_state['player_features']['TYPE_WICKETKEEPER'] == 1].reset_index()
        all_rounders = st.session_state['player_features'][st.session_state['player_features']['TYPE_ALL-ROUNDER'] == 1].reset_index()


        # Select number of each player type
        slider_col = st.columns([1, 1, 1])

        num_batsmen = slider_col[0].slider("Number of Batsmen", 1, 11)
        num_bowlers = slider_col[1].slider("Number of Bowlers", 1, 11-num_batsmen)
        num_wicket_keepers = slider_col[2].slider("Number of Wicket Keepers", 1, 11-num_batsmen-num_bowlers)
        num_all_rounders = 11 - num_batsmen - num_bowlers - num_wicket_keepers

        # st.write(st.session_state['player_features_scaled'])

        # Show dropdown for selecting players based on type
        batsman_type = slider_col[0].selectbox('Choose Batsman Type: ', options = batsmen['PLAYER'].unique())
        bowler_type = slider_col[1].selectbox('Choose Bowler Type: ', options = bowlers['PLAYER'].unique())
        wicket_keepers_type = slider_col[2].selectbox('Choose Wicket Keepers Type: ', options = wicket_keepers['PLAYER'].unique())
        all_rounders_type = st.selectbox('Choose All Rounder Type: ', options = all_rounders['PLAYER'].unique())

        recommend_player_button = st.button('Recommend Players >>')

        if recommend_player_button:
            selected_batsmen = recommend_players(batsman_type, n = num_batsmen, player_type = 'BATTER', player_features_scaled = st.session_state['player_features_scaled'])
            selected_bowlers = recommend_players(bowler_type, n = num_bowlers, player_type = 'BOWLER', player_features_scaled = st.session_state['player_features_scaled'])
            selected_wicket_keepers = recommend_players(wicket_keepers_type, n = num_wicket_keepers, player_type = 'WICKETKEEPER', player_features_scaled = st.session_state['player_features_scaled'])
            num_all_rounders = int(11 - (selected_batsmen.shape[0] + selected_bowlers.shape[0] + selected_wicket_keepers.shape[0]))
            selected_all_rounders = recommend_players(all_rounders_type, n = num_all_rounders, player_features_scaled = st.session_state['player_features_scaled'])

            
            selected_players = pd.concat([selected_batsmen, selected_bowlers, selected_wicket_keepers, selected_all_rounders])

            st.success("Team Selected Successfully!")
            st.session_state['selected_players'] = selected_players
            col_players = st.columns([1, 1])
            i = 0
            for player in selected_players.index:
                player_expander = col_players[i%2].expander(player)
                player_expander.write(st.session_state['player_df'][st.session_state['player_df']['PLAYER'] == player])
                i += 1

        
    if analysis_type == 'Network Analysis':
        st.title('Network Analysis')
        if 'selected_players' not in st.session_state:
            st.warning('Please make your team first to do perform the network analysis !!')

        else:
            chosen_player = st.selectbox('Choose the player for analysis: ', options = st.session_state['selected_players'].index)
            # Filter data for chosen_player
            player_data = st.session_state['df_combined']
            chosen_player_data_full = player_data[player_data['batsman'] == chosen_player]

            # Grouping by bowler to get total runs scored by player against each bowler
            bowler_runs = chosen_player_data_full.groupby('bowler')['batsman_runs'].sum().sort_values(ascending=False)

            # Grouping by bowler to get total balls faced by player against each bowler
            bowler_balls = chosen_player_data_full.groupby('bowler').size().sort_values(ascending=False)

            # Picking top 10 bowlers against whom player scored the most runs
            top_bowlers = bowler_runs.head(10)

            net_graph_1 = st.expander(f'Batsman-Bowler Network for {chosen_player}')
            cols_net_graph_1 = net_graph_1.columns([2, 1])

            # Creating a network graph
            G = nx.Graph()

            # Adding chosen_player as a node
            G.add_node(f'{chosen_player}', type='batsman', color='skyblue')

            # Adding top 10 bowlers chosen_player faced as nodes and edges with weight as total runs scored against them
            for bowler, runs in top_bowlers.items():
                G.add_node(bowler, type='bowler', color='red', size=bowler_balls[bowler])
                G.add_edge(f'{chosen_player}', bowler, weight=runs*5)

            # Network statistics: Degree Centrality
            degree_centrality = nx.degree_centrality(G)

            G.nodes[chosen_player]['size'] = 200

            # Plotting the network graph
            fig, axes = plt.subplots(figsize=(12, 12))
            colors = [G.nodes[node]['color'] for node in G.nodes]
            pos = nx.spring_layout(G)  # positions for all nodes

            # Use the number of balls faced to scale the node sizes
            node_sizes = [G.nodes[node].get('size', 1) * 50 for node in G.nodes]

            nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=node_sizes, ax=axes)
            nx.draw_networkx_edges(G, pos, width=[G[u][v]['weight'] * 0.02 for u, v in G.edges()], ax=axes)
            nx.draw_networkx_labels(G, pos, font_size=12, ax=axes)
            axes.set_title(f"Batsman-Bowler Network for {chosen_player}")
            plt.axis('off')
            cols_net_graph_1[0].pyplot(fig)

            degree = G.degree[chosen_player]  # Degree of the chosen_player node
            num_bowlers = len(G.nodes) - 1  # Number of bowlers in the graph (-1 to exclude the batsman)
            most_bowler = bowler_balls.idxmax()  # Bowler against whom the batsman faced the most balls
            most_run_bowler = bowler_runs.idxmax()  # Bowler against whom the batsman scored the most runs

            # Formatting the statistics
            stats = f"""
            - Degree of the {chosen_player} node is {degree}
            - Number of bowlers {chosen_player} played with is {num_bowlers}
            - The bowler bowled most of the time is {most_bowler}
            - {chosen_player} made most runs while playing against {most_run_bowler}
            """

            cols_net_graph_1[1].markdown('')
            cols_net_graph_1[1].markdown(f'### Description of Batsman-Bowler Network for {chosen_player}')
            cols_net_graph_1[1].write(stats)





            net_graph_2 = st.expander(f'Non-Striker Network for {chosen_player}')
            cols_net_graph_2 = net_graph_2.columns([2, 1])
            # Filtering rows where f'{chosen_player}' is the non_striker
            chosen_player_data = player_data[player_data['non_striker'] == f'{chosen_player}']

            # Grouping by batsman to get total runs made by the batsman across all matches
            total_runs_per_batsman = chosen_player_data.groupby('batsman')['id', 'batsman_runs'].sum().sort_values(ascending = False, by = 'batsman_runs').reset_index()

            # Create the network graph with the updated requirements
            G = nx.Graph()

            # Add {chosen_player} as central node
            G.add_node(f'{chosen_player}', size=0)

            # Add nodes and edges from the original data (not aggregated)
            for _, row in total_runs_per_batsman.iloc[:50, :].iterrows():
                # st.write(row)
                batsman = row['batsman']
                runs = row['batsman_runs']
                match_id = row['id']

                if not G.has_node(batsman):
                    total_runs = total_runs_per_batsman[total_runs_per_batsman['batsman'] == batsman]['batsman_runs'].values[0]
                    G.add_node(row['batsman'], size=total_runs, runs=total_runs)
                    
                edge_label = f"Match ID: {match_id}"
                G.add_edge(f'{chosen_player}', row['batsman'], label=edge_label, match_id=match_id)

            # Drawing the graph
            pos = nx.spring_layout(G)
            node_sizes = [G.nodes[node]['size'] * 20 + 50 for node in G.nodes()]
            node_labels = {node: node if node == f'{chosen_player}' else f"{node} ({G.nodes[node]['runs']})" for node in G.nodes()}
            edge_labels = nx.get_edge_attributes(G, 'label')

            fig1, ax = plt.subplots(figsize=(18, 12))
            nx.draw(G, pos, with_labels=True, labels=node_labels, node_color='skyblue', font_size=9, font_weight='bold', node_size=node_sizes, ax=ax)
            # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=ax, alpha=0.6)
            plt.title(f"{chosen_player}'s Partnerships as Non-Striker")

            cols_net_graph_2[0].pyplot(fig1)
            cols_net_graph_2[1].markdown('')

            # Total number of batsmen
            total_batsmen = total_runs_per_batsman.shape[0] #len(G.nodes()) - 1  # Subtracting one for {chosen_player}

            # Batsman with most runs and the runs
            # Extracting runs data for all players except the chosen player
            runs_data = [(data, node) for node, data in total_runs_per_batsman[['batsman', 'batsman_runs']].values if node != f'{chosen_player}']

            # Check if the list is empty
            if runs_data:
                max_runs_batsman = max(runs_data)
            else:
                max_runs_batsman = (0, None)  # Default value

            batsman_most_runs, most_runs = max_runs_batsman[1], max_runs_batsman[0]

            if runs_data:
                min_runs_batsman = min(runs_data)
            else:
                min_runs_batsman = (0, None)
            # min_runs_batsman = min(runs_data)
            batsman_least_runs, least_runs = min_runs_batsman[1], min_runs_batsman[0]

            # Average runs
            average_runs = sum([data['runs'] for node, data in G.nodes(data=True) if node != f'{chosen_player}']) / (total_batsmen + 0.000000001)

            # Total number of matches
            total_matches = len(set(nx.get_edge_attributes(G, 'match_id').values()))

            # Constructing the stats in f-string format
            stats = f"""
            - Total number of batsmen who batted with {chosen_player} as a non-striker: {total_batsmen}
            - Batsman who scored the most runs with {chosen_player} as non-striker: {batsman_most_runs} ({most_runs} runs)
            - Batsman who scored the least runs with {chosen_player} as non-striker: {batsman_least_runs} ({least_runs} runs)
            - Average runs scored by batsmen with {chosen_player} as non-striker: {average_runs:.2f} runs
            - Total number of matches represented in the graph: {total_matches}
            """


            # stats_1 = chosen_player_data.describe()
            cols_net_graph_2[1].markdown(f'### Description of network {chosen_player} as Non-Strikers')
            cols_net_graph_2[1].write(stats)




            net_graph_3 = st.expander(f'{chosen_player} with Other Non-Striker Players')
            cols_net_graph_3 = net_graph_3.columns([2, 1])
            # Grouping by non_striker to get total runs scored by player with each non_striker
            kohli_data_v2 = player_data[player_data['batsman'] == chosen_player].groupby(['non_striker']).sum().reset_index()

            # Grouping to get the number of times chosen_player played with each non_striker
            non_striker_counts = player_data[player_data['batsman'] == chosen_player]['non_striker'].value_counts().to_dict()

            # Create a network graph for chosen_player with other non-striker players
            G2 = nx.Graph()

            # Add edges for each non-striker chosen_player played with
            for non_striker, runs in zip(kohli_data_v2['non_striker'], kohli_data_v2['batsman_runs']):
                if non_striker != chosen_player:
                    G2.add_edge(chosen_player, non_striker, weight=non_striker_counts[non_striker])
                    G2.nodes[non_striker]['size'] = runs

            # Setting a fixed size for the chosen_player node
            G2.nodes[chosen_player]['size'] =300

            labels_2 = {node: f"{node} ({non_striker_counts[node]})" for node in G2.nodes() if node != chosen_player}
            color_map_2 = ['red' if node == chosen_player else 'skyblue' for node in G2.nodes()]
            pos_2 = nx.circular_layout(G2)
            fig2, ax2 = plt.subplots(figsize=(12, 7))
            node_sizes = [G2.nodes[node].get('size', 1) * 3 for node in G2.nodes()]
            # edge_widths = [G2[u][v]['weight'] * 0.05 for u, v in G2.edges()]

            nx.draw_networkx_nodes(G2, pos_2, node_size=node_sizes, node_color=color_map_2)
            nx.draw_networkx_edges(G2, pos_2)
            nx.draw_networkx_labels(G2, pos_2, labels=labels_2, font_size=8)
            plt.title(f"{chosen_player} with Other Non-Striker Players")
            plt.legend([chosen_player], loc='upper left')
            plt.axis("off")
            cols_net_graph_3[0].pyplot(fig2)
            cols_net_graph_3[1].markdown('')

            # Stats about runs made by non-strikers when chosen_player was batting
            # stats_1 = non_striker_runs.describe()
            cols_net_graph_3[1].markdown(f'### Description of network {chosen_player} with Other Non-Striker Players')
            # Calculate the statistics for the G2 graph

            # Total number of non-strikers
            total_non_strikers = len(G2.nodes()) - 1  # Subtracting one for chosen_player

            # Non-striker with whom chosen_player has played the most number of times
            max_matches_non_striker = max([(count, non_striker) for non_striker, count in non_striker_counts.items() if non_striker != f'{chosen_player}'])
            non_striker_most_matches, most_matches = max_matches_non_striker[1], max_matches_non_striker[0]

            # Non-striker with whom chosen_player has played the least number of times
            min_matches_non_striker = min([(count, non_striker) for non_striker, count in non_striker_counts.items() if non_striker != f'{chosen_player}'])
            non_striker_least_matches, least_matches = min_matches_non_striker[1], min_matches_non_striker[0]

            # Average number of matches with non-strikers
            average_matches = sum([count for non_striker, count in non_striker_counts.items() if non_striker != f'{chosen_player}']) / (total_non_strikers + 0.000000001)

            # Total number of matches represented in the graph - this is the sum of all matches with non-strikers
            total_matches_G2 = sum([count for non_striker, count in non_striker_counts.items() if non_striker != f'{chosen_player}'])

            # Constructing the stats in f-string format for G2 graph
            stats_G2 = f"""
            - Total number of non-strikers with whom {chosen_player} has played: {total_non_strikers}
            - Non-striker with whom {chosen_player} has played the most number of times: {non_striker_most_matches} ({most_matches} times)
            - Non-striker with whom {chosen_player} has played the least number of times: {non_striker_least_matches} ({least_matches} times)
            - Average number of times {chosen_player} has played with different non-strikers: {average_matches:.2f} times
            - Total number of matches represented in the graph: {total_matches_G2}"""

            cols_net_graph_3[1].write(stats_G2)






            net_graph_4 = st.expander(f'network where {chosen_player}"s Team Wins')
            cols_net_graph_4 = net_graph_4.columns([2, 1])
            # Filtering the data for only rows where the chosen_player played
            player_team_matches = player_data[(player_data['batsman'] == chosen_player) & 
                                            (player_data['batting_team'] == player_data['winner'])]

            # Counting the number of times each team won with the chosen_player
            team_wins_with_player = player_team_matches['batting_team'].value_counts().to_dict()

            # Creating the network graph
            G3 = nx.Graph()
            G3.add_node(chosen_player, type='player', color='red')

            # Adding edges for each team the chosen_player played with
            for team, wins in team_wins_with_player.items():
                G3.add_node(team, type='team', color='skyblue', size=wins)
                G3.add_edge(chosen_player, team, weight=wins)

            # Plotting the graph
            fig3, ax3 = plt.subplots(figsize=(12, 7))
            node_sizes = [G3.nodes[node].get('size', 1) * 200 for node in G3.nodes()]
            edge_widths = [G3[u][v]['weight'] * 0.1 for u, v in G3.edges()]
            color_map_3 = ['red' if node == chosen_player else 'skyblue' for node in G3.nodes()]
            pos_3 = nx.spring_layout(G3)

            nx.draw_networkx_nodes(G3, pos_3, node_size=node_sizes, node_color=color_map_3)
            nx.draw_networkx_edges(G3, pos_3, width=edge_widths)
            nx.draw_networkx_labels(G3, pos_3, font_size=8)
            plt.title(f"{chosen_player}'s Team Wins")
            plt.axis("off")
            cols_net_graph_4[0].pyplot(fig3)
            cols_net_graph_4[1].markdown('')
            # Extracting information for the description
            total_teams = len(team_wins_with_player)
            most_winning_team = max(team_wins_with_player, key=team_wins_with_player.get)
            most_wins = team_wins_with_player[most_winning_team]

            # Formatting the description using f-string
            description = f"""
            In the network graph:
            - {chosen_player} is connected to {total_teams} teams, representing each team he played for and won matches.
            - The team with which {chosen_player} has the most wins is {most_winning_team}, with a total of {most_wins} wins.
            - The size of each team node represents the number of wins with {chosen_player}.
            - The edge width indicates the number of wins for each team with {chosen_player}.
            """

            cols_net_graph_4[1].markdown(f'### Description of network where {chosen_player}"s Team Wins')
            cols_net_graph_4[1].markdown(description)


            # Performance statistics
            perf_stat_1 = st.expander('Performance statistics')
            # st.session_state['player_features'].to_csv('player_features_v1.csv')

            # Calculate relevant statistics
            total_runs = chosen_player_data_full['batsman_runs'].sum()
            total_matches = chosen_player_data_full['id'].nunique()
            total_innings = len(chosen_player_data_full.groupby(['id', 'inning']))
            average_runs = total_runs / total_innings

            # Data for plotting
            attributes = ['Total Runs', 'Matches Played', 'Innings', 'Average Runs/Inning']
            values = [total_runs, total_matches, total_innings, average_runs]

            # Create bar chart
            fig, axes = plt.subplots(figsize=(10, 6))
            axes.bar(attributes, values, color=['skyblue', 'green', 'red', 'purple'])
            axes.set_title(f'{chosen_player} Performance Statistics')
            axes.set_ylabel('Value')
            axes.set_xlabel('Attributes')

            # Display the plot
            plt.tight_layout()

            perf_stat_1.pyplot(fig)


            perf_stat_2 = st.expander(f'Top 10 bowlers against whom {chosen_player} scored the most runs')
            # Plotting the data
            fig, axes = plt.subplots(figsize=(12, 7))
            top_bowlers.plot(kind='bar', color='cyan', ax = axes)
            axes.set_title(f'Top 10 Bowlers Against Whom {chosen_player} Scored the Most Runs')
            axes.set_ylabel('Total Runs Scored')
            axes.set_xlabel('Bowlers')
            plt.xticks(rotation=45)
            plt.tight_layout()
            perf_stat_2.pyplot(fig)


            perf_stat_3 = st.expander(f'Top 10 Batsmen Who Played Alongside {chosen_player} as Non-Strikers')
            # Grouping by non_striker to get the frequency of each non_striker when chosen_player was batting
            non_strikers = chosen_player_data_full['non_striker'].value_counts().head(10)

            # Plotting the data
            fig, axes = plt.subplots(figsize=(12, 7))
            non_strikers.plot(kind='bar', color='orange', ax = axes)
            axes.set_title(f'Top 10 Batsmen Who Played Alongside {chosen_player} as Non-Strikers')
            axes.set_ylabel('Number of Balls')
            axes.set_xlabel('Batsmen')
            plt.xticks(rotation=45)
            plt.tight_layout()
            perf_stat_3.pyplot(fig)


            perf_stat_4 = st.expander(f'Runs Scored by {chosen_player} Against Different Teams')
            # Grouping by opposing team (team2 since chosen_player plays for Royal Challengers Bangalore, the dataset might have variations, so we consider both team1 and team2)
            team_runs = chosen_player_data_full.groupby('team2')['batsman_runs'].sum()

            # If chosen_player's team is listed as 'team2', then we should account for 'team1' as the opposing team
            team_runs_opposite = chosen_player_data_full.groupby('team1')['batsman_runs'].sum()
            team_runs = team_runs.add(team_runs_opposite, fill_value=0).sort_values(ascending=False)

            # Plotting the data
            fig, axes = plt.subplots(figsize=(12, 7))
            team_runs.plot(kind='bar', color='magenta', ax = axes)
            axes.set_title(f'Runs Scored by {chosen_player} Against Different Teams')
            axes.set_ylabel('Total Runs Scored')
            axes.set_xlabel('Teams')
            plt.xticks(rotation=90)
            plt.tight_layout()
            perf_stat_4.pyplot(fig)


    if analysis_type == 'Team Statistics':
        st.title("Team Statistics graph")
        player_data = st.session_state['df_combined']
        # Converting the 'date' column to datetime format
        player_data['date'] = pd.to_datetime(player_data['date'])
        # Extracting the year from the date
        player_data['year'] = player_data['date'].dt.year


        selected_teams = st.multiselect('Select Teams to Analyse', options = player_data['team2'].unique())
        selected_year = st.selectbox('Select Year', options = player_data['year'].unique())

        view_graph = st.button('View Graph')

        if view_graph:
            # Filtering the dataset for the selected teams and year
            filtered_data = player_data[(player_data['batting_team'].isin(selected_teams)) & 
                                        (player_data['bowling_team'].isin(selected_teams)) & 
                                        (player_data['year'] == selected_year)]

            # Color mapping for player types
            color_map_dict = {
                'batsman': 'gold',
                'non_striker': 'silver',
                'bowler': 'green',
                'fielder': 'blue',
                'umpire': 'red'
            }

            # Updating the graph based on the player relationships and different colors
            G4 = nx.Graph()

            for idx, row in filtered_data.iterrows():
                player_types = ['batsman', 'non_striker', 'bowler', 'fielder']
                
                for p_type in player_types:
                    player = row[p_type]
                    if pd.notna(player):  # Checking if the player field is not null
                        G4.add_node(player, type=p_type, color=color_map_dict[p_type])

                umpires = [row['umpire1'], row['umpire2']]
                for umpire in umpires:
                    if pd.notna(umpire):  # Checking if the umpire field is not null
                        G4.add_node(umpire, type='umpire', color=color_map_dict['umpire'])
                        for player in player_types:
                            G4.add_edge(row[player], umpire, label=umpire)

            # Extracting the color map for nodes
            color_map_4 = [G4.nodes[node].get('color', 'black') for node in G4.nodes()]
            pos_4 = nx.circular_layout(G4)

            # Plotting the graph with legend
            fig4, ax4 = plt.subplots(figsize=(12, 7))

            nx.draw_networkx_nodes(G4, pos_4, node_color=color_map_4, node_size=500)
            nx.draw_networkx_edges(G4, pos_4)
            nx.draw_networkx_labels(G4, pos_4, font_size=8)

            # Adding legend
            legend_labels = {v: k for k, v in color_map_dict.items()}
            patches = [plt.Line2D([0], [0], marker='o', color='w', label=legend_labels[color], 
                                markersize=10, markerfacecolor=color) for color in color_map_dict.values()]
            plt.legend(handles=patches, loc='upper right')

            plt.title(f"Player Relationships for {selected_teams} in {selected_year}")
            plt.axis("off")
            st.pyplot(fig4)

    if analysis_type == 'Player Value':
        st.title("Player Value Graphs")
        player_data = st.session_state['df_combined']

        # For demonstration purposes, selecting the top 11 bowlers based on the number of balls delivered
        selected_players = st.session_state['selected_players'].index.tolist()

        # Filtering the dataset for selected players as bowlers
        filtered_data = player_data[player_data['bowler'].isin(selected_players)]

        # st.write(filtered_data.bowler.unique())

        # Calculating number of balls delivered by each of the selected players
        balls_delivered = filtered_data['bowler'].value_counts().to_dict()

        # Calculating number of wickets taken by each bowler against every batsman
        # We'll consider player_dismissed as wickets (excluding run outs)
        wickets_data = filtered_data[filtered_data['dismissal_kind'].notna() & 
                                    (filtered_data['dismissal_kind'] != 'run out')]
        wickets_taken = wickets_data.groupby(['bowler', 'player_dismissed']).size().to_dict()

        # Creating the network graph
        G5 = nx.Graph()

        # Adding nodes for the selected players and edges based on wickets taken
        for (bowler, batsman), wickets in wickets_taken.items():
            G5.add_node(bowler, type='bowler', color='red', size=balls_delivered[bowler])
            G5.add_node(batsman, type='batsman', color='skyblue')
            G5.add_edge(bowler, batsman, weight=wickets)

        # Plotting the graph
        fig5, ax5 = plt.subplots(figsize=(14, 10))
        color_map_5 = [G5.nodes[node].get('color', 'green') for node in G5.nodes()]
        node_sizes = [G5.nodes[node].get('size', 1) * 0.5 for node in G5.nodes()]
        edge_widths = [G5[u][v]['weight'] * 0.5 for u, v in G5.edges()]
        pos_5 = nx.circular_layout(G5)

        nx.draw_networkx_nodes(G5, pos_5, node_color=color_map_5, node_size=node_sizes)
        nx.draw_networkx_edges(G5, pos_5, width=edge_widths)
        nx.draw_networkx_labels(G5, pos_5, font_size=8)
        plt.title(f"Relationship of Selected Bowlers with Batsmen")
        plt.axis("off")
        st.pyplot(fig5)

        # Extracting information for the description
        total_bowlers = len(balls_delivered)
        most_balls_bowler = max(balls_delivered, key=balls_delivered.get)
        most_balls = balls_delivered[most_balls_bowler]

        # Finding the bowler with the most wickets and the batsman against whom he took the most wickets
        max_wickets = 0
        max_wickets_bowler = ""
        max_wickets_batsman = ""
        for (bowler, batsman), wickets in wickets_taken.items():
            if wickets > max_wickets:
                max_wickets = wickets
                max_wickets_bowler = bowler
                max_wickets_batsman = batsman

        # Formatting the description using f-string
        description = f"""
        In the network graph:
        - There are {total_bowlers} selected bowlers, shown as red nodes.
        - The size of each bowler's node indicates the number of balls they've delivered.
        - {most_balls_bowler} has delivered the most balls, totaling {most_balls}.
        - Edges connect bowlers to batsmen they've dismissed.
        - The width of each edge signifies the number of times the bowler has dismissed the respective batsman.
        - {max_wickets_bowler} has taken the most wickets (total of {max_wickets}) against {max_wickets_batsman}.
        """

        st.markdown(description)