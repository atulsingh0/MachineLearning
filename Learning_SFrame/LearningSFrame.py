################################################################
##  Learning SFrame
##  Atul Singh
##  www.datagenx.net
################################################################


# import the graphlab
import graphlab as gl

# reading the csv file
data = gl.SFrame.read_csv("C:\learn\ML\ML00caseStudy\week01Intro\song_data.csv")

# getting no of rows and columns
data.num_rows(), data.num_columns() , len(data)

# saving the dataset
data.save('songs')

# loading the dataset
sdata = gl.load_sframe('songs')

# head 5 rows
print sdata.head(5)

# likewise tail 4 rows
print sdata.tail(4)

# checking the column names and data type
sdata.column_types(), sdata.column_names()


# ### Modify the data
# creating a new columns
sdata['my_rating'] = 0
print sdata

# generating a new colmns called song_age respective to my age
my_age = 1989
sdata['song_age']=sdata['year']-my_age
print sdata

# generating another column title length
sdata['title length'] = sdata['title'].apply(lambda x : len(x.split()))
print sdata

# adding one more column
sdata.add_column(sdata.select_column('year').apply(lambda x: x - my_age),name='how_old_was_i')
print sdata

# can add multiple columns at a time
sdata[['col1','col2']] = [sdata['title length'],sdata['my_rating']]
print sdata

#adding some more
#sdata[['col3','col4']] = [[3,4]]  this is not allowed, assigned values should be SArrray
#sdata

# deleting column
del sdata['song_age']

# rename columns
sdata.rename({'title length':'title_length'})
print sdata

# swap columns location
sdata.swap_columns('year','title_length')
print sdata

# change the column data type
sdata.column_names(),sdata.column_types()
sdata['my_rating'] = sdata['my_rating'].astype(float)
print sdata.column_types()

# swap column location
sdata.swap_columns('title_length','year')

# let's try to calculate how many 'a' are there in 'title', 'release' & 'artist_name' column
sdata.add_column(sdata['title', 'release' , 'artist_name'].apply(lambda row:sum(word.count('a') for word in row.values())),'no_of_a')
print sdata.head(3)


# ## Checking Missing values
# checking the missing year
year_count = sdata.groupby('year', gl.aggregate.COUNT)
print year_count.head(6)

# no of unique years
print "No of unique years :", str(len(year_count))
print "no of invalid year count "
print year_count.topk('year', reverse=True, k=1)

# year 0 is invalid value to better convert it to None
sdata['year'] = sdata['year'].apply(lambda x : None if x==0 else x)
print sdata.head(5)


# now I have to fix the value of 'How_old_Was_i' column as well
# I can do this two ways, one - substract - year - born_year 
# or I can use the same apply function with NONE if x < 0
sdata['how_old_was_i'] = sdata['year'].apply(lambda x : None if x is None else x-my_age)
print sdata.head(5)

# check the no of songs which have valid years
print len(sdata[sdata['year']>0])

tmp = sdata['year']>0
print tmp

# Look at lots of descriptive statistics of title_length
print "mean: " + str(sdata['title_length'].mean())
print "std: " + str(sdata['title_length'].std())
print "var: " + str(sdata['title_length'].var())
print "min: " + str(sdata['title_length'].min())
print "max: " + str(sdata['title_length'].max())
print "sum: " + str(sdata['title_length'].sum())
print "number of non-zero entries: " + str(sdata['title_length'].nnz())

approx_sketch = sdata['title_length'].sketch_summary()
print approx_sketch

# lets check which songs are having largest and smallest length
top_title_length = sdata.topk('title_length')
print top_title_length

# what about lowest
lowest_title_lenght = sdata.topk('title_length', reverse=True)
print lowest_title_lenght

# lowest 15
# what about lowest
lowest_title_lenght = sdata.topk('title_length', reverse=True, k =15)
print lowest_title_lenght
before_i_was_born = sdata['how_old_was_i'] < 0
before_i_was_born.all(), before_i_was_born.any()

# get total songs in an album and display the top album by no
sdata.groupby(['artist_name','release'], {'no_of_songs_in_album':gl.aggregate.COUNT} ).topk('no_of_songs_in_album')

# this will download the 118 MB file
#usage_data = gl.SFrame.read_csv("https://static.turi.com/datasets/millionsong/10000.txt", header=False, delimiter='\t', column_type_hints={'X3':int})
#usage_data.rename({'X1':'user_id', 'X2':'song_id', 'X3':'listen_count'})

# Read the data
usage_data = gl.SFrame.read_csv("C:/learn/ML/ML00caseStudy/week01Intro/10000.txt", header=False, delimiter="\t", column_type_hints={'X3':int})
usage_data.rename({'X1':'user_id', 'X2':'song_id', 'X3':'listen_count'})

# saving this data frame
usage_data.save('usage_data')

# loading  the data
usage_data = gl.load_sframe('usage_data')

# find out the unique users
print len(usage_data['user_id'].unique())

# let's create two datasets which we can join
ds1 = sdata[((sdata['artist_name'] == 'Relient K')
                           | (sdata['artist_name'] == 'Streetlight Manifesto'))
                          & (sdata['how_old_was_i'] >= 14) & (sdata['how_old_was_i'] <= 18)]
print ds1

# Let's join ds1 with the 10000.txt dataset usage_data
dsjoin = ds1.join(usage_data, 'song_id')
print dsjoin

# total row in ds1 and dsjoin datasets
print len(ds1), len(dsjoin)
print len(ds1['song_id'].unique()), len(dsjoin['song_id'].unique()), len(ds1['song_id']), len(dsjoin['song_id'])

# find out most popular songs when I was between 14 n 18
most_popular = dsjoin.groupby(['song_id'], {'total_listen_count':gl.aggregate.SUM('listen_count'), 
                                             'num_unique_users':gl.aggregate.COUNT('user_id')})
print most_popular

# to get artist name we have to join this data
most_popular.join(sdata, 'song_id').topk('total_listen_count',k=20)

# let's append a row with max liste count and check whether it comes in above result or not
me = gl.SFrame({'user_id':['evan'],'song_id':['SOSFAVU12A6D4FDC6A'],'listen_count':[4000]})

# adding this data to usage data
usage_data = usage_data.append(me)

# repeating the above join n group by statement
dsjoin = ds1.join(usage_data, 'song_id')
most_popular = dsjoin.groupby(['song_id'], {'total_listen_count':gl.aggregate.SUM('listen_count'), 
                                             'num_unique_users':gl.aggregate.COUNT('user_id')})
most_popular.join(sdata, 'song_id').topk('total_listen_count',k=20)



# ### Splitting and Sampling
# Lets check, how we can randomly split the data for test

# Randomly split data rows into two subsets
first_set, second_set = sdata.random_split(0.8, seed = 1)
print first_set.num_rows(), second_set.num_rows()

# If you want to split on a predicate though, you'll have to do that manually.
songs_before = sdata[sdata['how_old_was_i'] < 0]
songs_after = sdata[sdata['how_old_was_i'] >= 0]
print songs_before.num_rows(), songs_after.num_rows()

# generating sample data
sample = sdata.sample(0.4)
print sample.num_rows()


# ### SArray
arr = gl.SArray([1,2,3])
print arr

arr2 = 2*arr
print arr2

# add
print arr + arr2

# multiply
print arr * arr2

# divide
print arr2 / arr

# iterating with SFrame
for i in sdata:
    if i['title_length'] >= 45:
        print "Whoa that's long!"


# ### Using apply function on SFrame

sdata['title_artist_length'] = sdata['title','artist_name'].apply(lambda row: sum([len(col) for col in row.values()]))
print sdata


# ### Saving Our Work

# save as csv
sdata.save('sdata_new.csv', format='csv')
sdata.save('sdata_new')


############################################################
## Atul Singh  | www.datagenx.net | lnked.in/atulsingh
############################################################
