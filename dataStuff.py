import pandas as pd

# made ratings to a global variable, because I needed it in the main file (ugly but a quick fix)
ratings = {'completion', 'overall_ok', 'lost_adhesion', 'not_labelable', 'blobs', 'gaps', 'layer_misalignment',
               'layer_separation', 'over_extrusion', 'line_misalignment', 'stringing', 'under_extrusion', 'warping',
               'poor_bridging', 'burning'}


# get one hot dummies of column and return these
def one_hot_column(dataframe, column_name):
    one_hotted_columns = pd.get_dummies(dataframe, columns=[column_name], prefix=column_name)

    return one_hotted_columns


# convert bool to int columns, remove the bool column and return the dataframe with now int columns
def convert_bool_to_int_column(dataframe, column_name):
    column_as_int = dataframe[column_name].astype(int)
    dataframe = dataframe.drop(column_name, axis=1)
    dataframe = pd.concat([dataframe, column_as_int], axis=1)

    return dataframe


# do absolute maximum scale
def absolute_maximum_scale(series):
    return series / series.abs().max()


# main function to do all the data cleanup
def do_data_stuff():
    # read data in pandas dataframe
    df = pd.read_csv('data.csv')

    # drop unneeded ID and data ID columns
    df = df.drop(df.columns[[0, 1]], axis=1)

    # get all columns with relevant ratings and extract them into a seperate dataframe
    target_df = pd.concat([df.pop(rating) for rating in ratings], axis=1)

    # get all columns that needs to get one_hotted and do that
    to_one_hot = {'printer', 'adhesion_type', 'support_type', 'slicing_tolerance', 'material_color', 'material_type',
                  'material_producer'}
    for item in to_one_hot:
        df = one_hot_column(df, item)

    # get all columns that to convert booleans to int and do that
    to_convert_bool_to_int = {'retraction_hop_enabled', 'ironing_enabled', 'limit_support_retractions',
                              'support_enable', 'cool_fan_enabled', 'ironing_only_highest_layer', 'retraction_enable',
                              'bridge_settings_enabled', 'support_interface_enable', 'support_tree_enable'}
    for item in to_convert_bool_to_int:
        df = convert_bool_to_int_column(df, item)

    # scale all columns to a absolute maximum between 0 and 1
    for col in df.columns:
        df[col] = absolute_maximum_scale(df[col])

    # round all values of ratings to a full integer
    target_df = target_df.round(decimals=0)

    # write out the dataframes as csv files (I saved them to evaluate the result of my methods and debug if needed)
    df.to_csv('new_data.csv', encoding='utf-8', index=False)
    target_df.to_csv('new_data_ratings.csv', encoding='utf-8', index=False)

    return df, target_df
