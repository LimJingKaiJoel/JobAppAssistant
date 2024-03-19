import pandas as pd

# change this accordingly
df = pd.read_csv('./download_this_dataset/FTData.csv')

TITLE_COL = "Title"

# CHANGE THESE TO ADD NEW WORDS : if none of these words appeared, add to junior
intern_keywords = ['intern', 'internship']
junior_keywords = ['junior', 'entry']
senior_keywords = ['senior', 'manager', 'lead']
#pt_keywords = ['part-time', 'part time', 'part_time', 'parttime']

# count number of each keyword and find max
def categorize_row(row, column_name):
    # how many times each word appears
    intern_count = sum(row[column_name].lower().count(keyword) for keyword in intern_keywords)
    junior_count = sum(row[column_name].lower().count(keyword) for keyword in junior_keywords)
    senior_count = sum(row[column_name].lower().count(keyword) for keyword in senior_keywords)
    #pt_count = sum(row[column_name].lower().count(keyword) for keyword in pt_keywords)

    # most appeared word
    max_count = max(intern_count, junior_count, senior_count)

    if max_count == 0 or max_count == junior_count:
        return 'junior'
    elif max_count == intern_count:
        return 'intern'
    elif max_count == senior_count:
        return 'senior'
    # elif max_count == pt_count:
    #     return 'pt'
    else:
        return 'error'
    
if __name__ == "__main__":
    # categorise each row
    # print(df.columns)
    #print(df['Title'])

    # change column_name to your column name of the job title of your dataset
    df['category'] = df.apply(categorize_row, column_name=TITLE_COL, axis=1)

    # create 4 new df
    intern_df = df[df['category'] == 'intern']
    junior_df = df[df['category'] == 'junior']
    senior_df = df[df['category'] == 'senior']
    #pt_df = df[df['category'] == 'pt']

    # save each df to a new csv
    intern_df.to_csv('./download_this_dataset/separated_data/InternData.csv', index=False)
    junior_df.to_csv('./download_this_dataset/separated_data/EntryData.csv', index=False)
    senior_df.to_csv('./download_this_dataset/separated_data/SeniorData.csv', index=False)
    #pt_df.to_csv('./download_this_dataset/separated_data/PTData.csv', index=False)