import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.linalg import norm
import math
from IPython.display import Markdown as md
import matplotlib.patches as mpatches
from matplotlib import cm

# What I am using for each letter on the grid
# {"A": "0", "B": "1", "C": "2", "D": "3", "E": "4", "F": "5", "G": "6", "H": "7", "I": "8", "J": "9", "K": "10"}


def read_warzone(path):
    warzone = pd.read_csv('Warzone.csv', header=2, dtype={
                          "drop_area": "category"})
    warzone["date"] = pd.to_datetime(warzone["date"])
    return warzone


def create_distance_from_center_col(warzone):
    drop_center_dist = []

    for index, row in warzone.iterrows():
        drop = np.array([row['drop_location_lat'], row['drop_location_lon']])
        center = np.array([row['circle_middle_lat'], row['circle_middle_lon']])

        drop_center_dist.append(math.hypot(
            drop[0] - center[0], drop[1] - center[1]))

    return drop_center_dist


def create_distance_from_plane_col(warzone):
    drop_plane_dist = []

    for index, row in warzone.iterrows():
        drop = np.array([row['drop_location_lat'], row['drop_location_lon']])
        pl_st = np.array([row['plane_start_lat'], row['plane_start_lon']])
        pl_end = np.array([row['plane_end_lat'], row['plane_end_lon']])

        drop_plane_dist.append(
            np.abs(np.cross(pl_end-pl_st, pl_st-drop))/norm(pl_end-pl_st))

    return drop_plane_dist


def get_corr(df, col1, col2):
    return df[col1].corr(df[col2])


def get_multi_corr(df_len, x, y, z):
    Rxyz = math.sqrt((abs(x**2) + abs(y**2) - 2*x*y*z) / (1-abs(z**2)))
    R2 = Rxyz**2

    n = df_len  # Number of rows
    k = 2       # Number of independent variables
    R2_adj = 1 - (((1-R2)*(n-1)) / (n-k-1))

    return R2_adj


def get_season_averages(warzone):
    season_average_string = ''
    for season in warzone['season'].unique():
        season_average_string += "Season " + str(season) + " average: " + str(
            round(warzone.groupby('season').get_group(season)['place'].mean(), 2)) + "\n"
    return season_average_string


def plot_gametypes_by_number(warzone):
    plt.subplots(figsize=(10, 5), dpi=80)
    game_plot = sns.countplot(x="team_num", data=warzone)
    game_plot.set(xlabel='Game Type', ylabel='Number of Games')
    game_plot.set_xticklabels(["Duos", "Trios", "Quads"])


def print_average_places(duos_mean, trios_mean, quads_mean, season_average_string, overall_mean):
    print(season_average_string + "\nDuos Average: " + str(duos_mean) + "\nTrios Average: " +
          str(trios_mean) + "\nQuads Average: " + str(quads_mean) + "\n\nOverall our Average is: " + str(overall_mean))


def display_placement_by_drop_location_chart(warzone):
    drop_locations_counts = warzone['drop_area'].value_counts(ascending=True)
    bins = pd.IntervalIndex.from_tuples(
        [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 75)])
    colors = {'(0, 5]': '#B50727', '(5, 10]': '#E26951', '(10, 15]': '#F6A384', '(15, 20]': '#E8D6CC',
              '(20, 25]': '#BBD1F8', '(25, 30]': '#8FB1FE', '(30, 35]': '#5673E0', '(35, 75]': '#3D50C3'}

    sns.despine(left=True, bottom=True)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=80)
    ax.set_xlabel('Number of Games')

    for index, loc in enumerate(drop_locations_counts.index):
        place_by_area = warzone.loc[warzone['drop_area'] == loc, 'place']
        place_bins = pd.cut(place_by_area, bins).value_counts()
        position = 0
        for index, values in enumerate(place_bins):
            bin = place_bins.index[index]
            plt.barh(y=loc, width=values, left=position,
                     color=colors[str(bin)])
            position += values

    plt.legend(handles=create_drop_location_legend(
        colors), title='Place Finished')


def create_drop_location_legend(colors):
    color_handles = []
    for index, color in enumerate(colors.values()):
        if index == 7:
            color_handles.append(mpatches.Patch(color=color, label='36+'))
        else:
            color_handles.append(mpatches.Patch(
                color=color, label=str(index*5 + 1) + '-' + str(index*5 + 5)))
    return color_handles


def print_drop_locations_averages(drop_location_average_finish):
    place_str = "The average for the places we have been to so far:\n"
    for index, place in enumerate(drop_location_average_finish):
        place_str += drop_location_average_finish.index[index] + ": " + str(
            place) + "\n"
    print(place_str)


def plot_drop_location_heatmap(warzone):
    drop_coordinates = warzone[['drop_location_lat', 'drop_location_lon']]
    heatmap, xedges, yedges = np.histogram2d(drop_coordinates['drop_location_lat'], drop_coordinates['drop_location_lon'],
                                             bins=10, range=[[0, 10], [0, 10]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    im = plt.imread('Verdansk.png')
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(im, origin='lower', aspect='auto', extent=(0, 10, 10, 0))
    ax.imshow(heatmap.T, extent=extent, origin='upper', alpha=.5, cmap="magma")


def plot_distance_scatters(warzone):
    fig, ax = plt.subplots(1, 2, figsize=(20, 6), dpi=80)
    ax1 = sns.scatterplot(data=warzone, x='drop_plane_dist',
                          y='place', ax=ax[0], s=30)
    ax2 = sns.scatterplot(
        data=warzone, x='drop_center_dist', y='place', ax=ax[1], s=30)
    ax1.set_xlabel("Distance From Drop Location to Flight Path")
    ax1.set_ylabel("Place Finished")
    ax2.set_xlabel("Distance From Drop Location to Center of Circle")
    ax2.set_ylabel("Place Finished")


def print_distance_correlation_string(place_plane_corr_form, place_center_corr_form):
    print("The correlation between the distance from drop location to plane and place is: " +
          "{:.2%}\nThe correlation between the distance from drop location to the center of".format(place_plane_corr_form) +
          "the circle and place is:{:.2%}\nSo far not really any correlation between these".format(place_center_corr_form) +
          "values. Let's try taking the correlationbetween both distances and how well we did.")

def plot_correlation_both_dist_and_place(warzone):
    fig, ax = plt.subplots(figsize=(10,5), dpi=80)
    ax1 = sns.scatterplot(data=warzone, x='drop_plane_dist', y='drop_center_dist', 
                      size=warzone['place'], hue=warzone['place'], sizes=(10,300), palette='viridis')
    ax1.set_title("Correlation between the two distances")

def print_correlation_both_dist_and_place(corr_between_place_both_dists):
    print("The size of the circles represent how badly we did.\n" +
    "As in the bigger the circle the worse we did. And as you can see" +
    "there seems be no correlation between placement and both distances," +
    "the value being: {:.2%}\n Lets try summing the distances and taking".format(corr_between_place_both_dists) +
    "that correlation.")

def print_combined_dist_string(combined_dist_corr):
    print("This time the correlation between combined distances and place is {:.2%}".format(combined_dist_corr) +
    "\nSo once again very little correlation, we will have to collect more data, to see if this changes.")

def plot_first_place_finishes(warzone):
    top_warzone_place = warzone[warzone['place'] == 1]
    colors = cm.get_cmap('tab20').colors[:len(top_warzone_place.index)]
    top_warzone_place.index = np.arange(len(top_warzone_place.index))
    im = plt.imread('Verdansk.png')
    fig, ax = plt.subplots(figsize=(20,20))
    ax.imshow(im, origin='upper', aspect='auto', extent=(0,10,10,0))
    ax.scatter(x='circle_middle_lat', y='circle_middle_lon', data=top_warzone_place, color=colors, s=300)
    ax.scatter(x='drop_location_lat', y='drop_location_lon', data=top_warzone_place, color=colors, marker='*', s=300)
    ax.legend()

    for index, row in top_warzone_place.iterrows():
        plt.plot(row[["plane_start_lat", "plane_end_lat"]],row[["plane_start_lon", "plane_end_lon"]], color=colors[index], lw=2)

    sns.despine(bottom=True, left=True)