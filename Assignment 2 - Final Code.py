#!/usr/bin/env python
# coding: utf-8

# In[8]:


from scipy.stats import pearsonr
import sys
import pandas as pd
import numpy as np
import os

print("\014")  # Clear Screen
# In[9]:


current_directory = os.getcwd()
print(current_directory)


# In[10]:


file_path = "u.data"


# In[11]:


separator = '\t'  # Replace with the actual separator used in your file

# Read the file into a DataFrame
df = pd.read_csv(file_path, sep=separator, header=None)
# Set the column names
df.columns = ['user id', 'movie id', 'rating', 'meta data']

df.head()


# In[12]:


num_rows = df.shape[0]
print(f"\nThe data file contains {num_rows} rows.")

# Drop the last column
df.drop(df.columns[-1], axis=1, inplace=True)
df.head()


# In[13]:


df2 = df.copy()


# In[14]:


pivot_df = df2.pivot(index='user id', columns='movie id', values=['rating'])
# Replace NaN with 0 in the entire DataFrame
pivot_df_filled = pivot_df.fillna(0)


# In[15]:
print('\n\nGetting three users as the group of users')
print('---------------------------------------------')
print('\n\nEnter numbers between 0 - 942')
print('---------------------------------------------')
try:
    user_input_1 = int(input("Enter the first row index: "))
    user_input_2 = int(input("Enter the second row index: "))
    user_input_3 = int(input("Enter the third row index: "))
except ValueError:
    print("Please enter valid integers.")
    exit()


# In[16]:

print('\n\nSelect Rows based on user input')
# Select rows based on user input
selected_rows = pivot_df_filled.iloc[[
    user_input_1, user_input_2, user_input_3]]

# Display the selected rows
print("\nSelected Rows:")
print(selected_rows)
# Create a new DataFrame from selected rows
new_dataframe = pd.DataFrame(selected_rows)
new_matrix = new_dataframe.values


# In[17]:

print('\nTo Calculate the Pearson Corelation we need movies rated by both users. select_Items_ratedByBoth" function does that ')
# To Calculate the Pearson Corelation we need movies rated by both users.
# "select_Items_ratedByBoth" function does that.


def select_Items_ratedByBoth(new_dataframe, user1, user2):
    #num_rows, num_columns = pivot_df_filled.shape
    user1Row = []
    user2Row = []
    x = new_dataframe.iloc[user1].to_numpy()
    y = new_dataframe.iloc[user2].to_numpy()
    common_non_zero_positions = np.where((x != 0) & (y != 0))[0]

    # print(common_non_zero_positions)
    user1Row = x[common_non_zero_positions]
    user2Row = y[common_non_zero_positions]

    return user1Row, user2Row


# In[18]:


def find_similariry_between_users(new_dataframe):
    num_people, num_movies = new_dataframe.shape
    # make a zero similarity matrtix
    similarity_matrix = np.zeros((num_people, num_people))
    for i in range(0, num_people):
        for j in range(0, num_people):
            if(i != j):
                user1Row, user2Row = select_Items_ratedByBoth(
                    new_dataframe, i, j)
                if (len(user1Row) <= 3 or np.std(user1Row) == 0 or np.std(user2Row) == 0):
                    similarity_matrix[i, j] = 0
                else:
                    correlation_coefficient, p_value = pearsonr(
                        user1Row, user2Row)
                    similarity_matrix[i, j] = correlation_coefficient

    # Replace NaN values with zeros
    similarity_matrix = np.nan_to_num(similarity_matrix, nan=0.0)
    return similarity_matrix


# In[19]:

print('\nFinding similarities between_users')
similarity_matrix = find_similariry_between_users(new_dataframe)
print(similarity_matrix)


# In[20]:

print('\nSince there are three users, to calculate recomedations we select the best user for each user')
top1_indices = np.argsort(similarity_matrix, axis=1)[:, -1:][:, ::-1]
top1_values = np.sort(similarity_matrix, axis=1)[:, -1:][:, ::-1]

print('Best user for each user\n', top1_indices)
print('Best users similarity value\n', top1_values)

# best matching user index for each user (3x1)matrix
new_df_1 = pd.DataFrame(top1_indices)
# best matching similarlity value for each user (3x1)matrix
new_df_2 = pd.DataFrame(top1_values)


# In[21]:


# we pass tot userxmovies rating matrix, top 1 similar user indexs mat and the top 1 sim value matrices
# We take a particular user, his best matching one users, and using those of his 1 users, for that particular user,
# we give a rating for each movie
# So here the output is 3X1682 matrix each having predictions for ech user for each movie, base on their one
# similar users
def find_prediction_for_user(new_dataframe, new_df_1, new_df_2):
    num_movies = 1682
    num_of_users = 1  # counted similar users
    num_of_All_users = 3  # all users are three

    predicted_movie_matrix = np.zeros((num_of_All_users, num_movies))

    for i in range(0, num_of_All_users):
        moviePredictionRate = []
        pure_user_A_Rate = [
            x for x in new_dataframe.iloc[i].to_numpy() if x != 0]
        mean_user_A_Rate = np.mean(pure_user_A_Rate)

        for j in range(0, num_movies):
            sumPartA = 0
            sumPartB = 0

            for k in range(0, num_of_users):
                userB_index = new_df_1.iloc[i, k]
                similarity_measure_AB = new_df_2.iloc[i, k]

                pure_user_B_Rate = [
                    x for x in new_dataframe.iloc[userB_index].to_numpy() if x != 0]
                mean_user_B_rate = np.mean(pure_user_B_Rate)
                rate_for_jth_movie = new_dataframe.iloc[userB_index, j]

                sumPartA = sumPartA + \
                    (similarity_measure_AB * (rate_for_jth_movie - mean_user_B_rate))
                sumPartB = sumPartB + similarity_measure_AB

            if sumPartB != 0:
                pred_jth_movie = mean_user_A_Rate + (sumPartA / sumPartB)
                #print('pred_jth_movie', pred_jth_movie)
            else:
                # Handle the case where sumPartB is zero (division by zero)
                pred_jth_movie = 0  # or set to another appropriate value

            moviePredictionRate.append(pred_jth_movie)

        predicted_movie_matrix[i, :] = moviePredictionRate

    return predicted_movie_matrix


# In[22]:
print('\nPredictions for each user based on its best mattching user')
predicted_movie_matrix = find_prediction_for_user(
    new_dataframe, new_df_1, new_df_2)
print(predicted_movie_matrix)


# In[23]:


# Movie predictions for all three users
movie_predictions_for_all_users = pd.DataFrame(predicted_movie_matrix)


# Now we are ready to apply the methods to find movie prediction to the group. Let r*(u, i) be either the predicted rating of u for i, or r(u, i) if this rating is present in the data set. So we need to make a marix, having predictions as well as if their is a rating from that user to th movie, we wil keep as it is. So we will create a nex matrix using the original rating matrix and the predicted_movie_matrix

# In[24]:

print('\nNow we are ready to apply the methods to find movie prediction to the group. Let r*(u, i) be either the predicted rating of u for i, or r(u, i) if this rating is present in the data set. So we need to make a marix, having predictions as well as if their is a rating from that user to th movie, we wil keep as it is. So we will create a new matrix using the original rating matrix and the predicted_movie_matrix')
Matrix1 = predicted_movie_matrix  # 3x1682 new rating matrix
Matrix2 = new_matrix  # 3x1682 original rating matrix

# Find the non-zero values and their positions in matrix2
non_zero_values = Matrix2[Matrix2.nonzero()]
non_zero_positions = np.nonzero(Matrix2)

# Replace the corresponding values in matrix1
Matrix1[non_zero_positions] = non_zero_values

# Convert Matrix 1 into a data frame
# This is the matrix that we apply group recomendation methods
new_pred_matrix_to_apply_aggrigation = pd.DataFrame(Matrix1)

# In[25]:


# The first aggregation approach is the average method.
# We take average value for each movie, sort them in decsending order
# and give movie prediction for the group as the movies with hogher average values
# We are showing top 10 movies that can be recomend to the 3 users using the average method
print('\nThe first aggregation approach is the average method.')
print('........................................................\n')
print('\nWe calculate average of ratings with only finite and non-negative values')


def average_non_negative_nonfinite(column):
    valid_values = column[(np.isfinite(column)) & (column >= 0)]
    return np.mean(valid_values)


# Calculate column averages without non-finite values
column_means = new_pred_matrix_to_apply_aggrigation.apply(
    average_non_negative_nonfinite)


# Find the column names of the ten columns with the maximum average
top_columns_mean = column_means.nlargest(10)


# Print the results
print("Original DataFrame:")
print(new_pred_matrix_to_apply_aggrigation)
print("\nAverage of Each Column:")
print(column_means)
print("\nTop 10 Columns with Maximum Average:")
print(top_columns_mean)

# Find the column names of the ten columns with the maximum average
top_columns_index = column_means.nlargest(10).index
print('Suggested movie IDS', top_columns_index+1)


# In[26]:


# The second aggregation approach is the least misery method.
# We take minimum value for each movie, sort them in decsending order
# and give movie prediction for the group as the movies with higher minimum values
# We are showing top 10 movies that can be recomend to the 3 users using the least misery method.

print('\nThe second aggregation approach is the least misery method.')
print('\n----------------------------------------------------------')
print('\nWe calculate minimum of ratings with only finite and non-negative values')
# Function to calculate minimum with only non-finite and non-negative values


def minimum_non_negative_nonfinite(column):
    valid_values = column[(np.isfinite(column)) & (column >= 0)]
    return np.min(valid_values)


# Calculate column averages without non-finite values
column_minimums = new_pred_matrix_to_apply_aggrigation.apply(
    minimum_non_negative_nonfinite)


# Find the column names of the ten columns with the maximum minimum
top_columns = column_minimums.nlargest(10)


# Print the results
print("Original DataFrame:")
print(new_pred_matrix_to_apply_aggrigation)
print("\nminimum of Each Column:")
print(column_minimums)
print("\nTop 10 Columns with Maximum minimum values:")
print(top_columns)

# Find the column names of the ten columns with the maximum minimums
top_columns_index = column_minimums.nlargest(10).index
print('Suggested movie IDS', top_columns_index+1)


# In[29]:
print('\n\n------------------Implementation - PART b--------------')
print('\n\nHere we need to define a way for counting the disagreements between the users in a group. So we used standard deviation of predicted ratings. Higher standard deviation implies more significant disagreement.')

# Function to calculate average with only finite and non-negative values


def average_non_negative_nonfinite(column):
    valid_values = column[(np.isfinite(column)) & (column >= 0)]
    return np.mean(valid_values)

# Function to calculate std with only finite and non-negative values


def std_non_negative_nonfinite(column):
    valid_values = column[(np.isfinite(column)) & (column >= 0)]
    return np.std(valid_values)


std_deviation = new_pred_matrix_to_apply_aggrigation.apply(
    std_non_negative_nonfinite)

print(std_deviation)

maxSTD = max(std_deviation)
minSTD = min(std_deviation)
# print(maxSTD)
# print(minSTD)

print('\n\nFix a threshold as 80% of the difference between the standard deviation for each column/movie')
threshold = round(0.5*(maxSTD-minSTD), 2)
print(threshold)

# Step 2: Identify columns with standard deviation less than threshold
selected_columns = std_deviation[std_deviation < threshold].index

# Step 3: Calculate and store the average for selected columns
average_values = new_pred_matrix_to_apply_aggrigation[selected_columns].apply(
    average_non_negative_nonfinite)

# Step 4: Find the indices of the top 10 columns with maximum average
top_10_indices = average_values.nlargest(10).index

# Step 4: Find the indices of the top 10 columns with maximum average
top_10_vals = average_values.nlargest(10)

# Display the results
print('Selected Columns with Standard Deviation < threshold:')
print(selected_columns)

print('\nAverage Values for Selected Columns:')
print(average_values)

print('\nTop 10 Movies with Maximum Average:')
print(top_10_indices+1)
print(top_10_vals)


# In[ ]:
