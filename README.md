# Playdate Community Census 2022

This repository contains the raw census data and various cleaned and tagged datasets. It also has my python notebook files that I used for processing and visualizing the data, although these are not really organized or structured enough to be used by someone other than me. But you are welcome to look at them anyway- just ask if you need help.

## Data Sets

- [raw_data/](raw_data/): Contains the raw csv export of the full sentence, plus a col_mapping.csv that shows how I've mapped the fully question descriptions to short column titles.
- [normalized_short_responses/](normalized_short_responses/): A "cleaned short response" is one that has been given a standardized label or has been marked as "Other". The only files that really matter are those which are suffixed by `_cleaned.csv`, the rest are just intermediary filesthat were used in the manual processing steps. The 'cleaned' CSVs have two colums: pdidx (the original id of the comment) and the short column name for the column that is cleaned with normalized responses. There are some rows with multiple repeating pdidx IDs: these are responses from a single individual that have been split into multiple rows, one for each selection. 
  - There are also two files called `exciting_non_s1.csv` and `exciting_s1.csv`. These are cleaned versinos of the "game" free responses, but they have a few more columns of data to help with filtering by those who own a playdate or have ordered a playdate.
- [tagged_open_responses/](tagged_open_responses/): These are the long-form responses that have been tagged by volunteers. The columns are each different tags, and rows are the responses. A response will have 1s in each column for tags it is associated with, 0s otherwise.
- [wordcloud_data/](wordcloud_data/): These are CSVs with weight and word columns to use for importing into wordclouds.com. The word list is calculated after pre-processing (removed punctionation, lowercase, add common bi_gram phrases, etc) and the weights for each word are an integer-scaled TF-IDF weight.

## Python Files

- `main_report.ipynb`: Notebook for data slicing and calling plotting functions
- `utils.py`: Custom plotting utility functions based around the Plotly library
- `normalized_short_responses.ipynb`: Scratch-pad like notebook for normalizing short free responses. This is the most incomprehensible of the bunch. 





