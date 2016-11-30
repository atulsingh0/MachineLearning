
# importing graphlab 
import graphlab as gl

# creating SFrame object by reading a csv file
data = gl.SFrame.read_csv("C:\learn\ML\ML00caseStudy\week01Intro\song_data.csv")

# view content of sf variable
data

# printing head 5 line, first 5 lines
data.head(5)

# likewise tail
data.tail(3)

# getting no of rows and columns
data.num_rows(), data.num_columns() , len(data)

# checking the column names and data type
sdata.column_types(), sdata.column_names()

# saving the dataset
data.save('songs')

# loading the dataset
sdata = gl.load_sframe('songs')

# creating a new columns
sdata['dummy'] = 'Atul'  
sdata['my_rating'] = 0    # this will populate 0 for all rows
# used when we are generating column with some default value
# or
sdata.add_column([value_array], 'my_rating2') #when we have diff values for each row
# usually used when we are generating new column from another column

# deleting column
del sdata['dummy']

# rename columns
sdata.rename({'title length':'title_length'})

# swap column location
sdata.swap_columns('title_length','year')

# change column data type
sdata['my_rating'] = sdata['my_rating'].astype(float)


