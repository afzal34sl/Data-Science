# importing the data_patterns module
import data_patterns

# importing the pandas and numpy module
import pandas as pd
import numpy as np
from io import StringIO
import re

# # reading the given csv file
# # and creating dataframe
df_list = []  # as you want these as your headers
check_scheme_name = 0
col_name = ''
i = 0
with open("Navaii MF.txt") as f:
    for line in f:
        # remove whitespace at the start and the newline at the end
        i = i + 1
        if line == 3: continue
        line = line.strip()
        find_semi_colon = line.find(';')
        if line != '':
            if check_scheme_name == 0:
                if find_semi_colon != -1:
                    columns = re.split(';', line)
                    if i == 1:
                        columns.insert(0, 'Company Name')
                    else:
                        columns.insert(0, col_name)
                    df_list.append(columns)
                else:
                    check_scheme_name = 1
                    col_name = line

            elif check_scheme_name == 1:
                # split each column on whitespace
                columns = re.split(';', line)
                columns.insert(0, col_name)
                df_list.append(columns)
        else:
            check_scheme_name = 0

    df_list

    df = pd.DataFrame(df_list)

    df.head()

    # store dataframe into csv file
    df.to_csv('NAVAII.csv', index=None)
