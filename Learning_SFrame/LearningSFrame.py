

# In[2]:
# import the graphlab
import graphlab as gl


# In[3]:
# reading the csv file
data = gl.SFrame.read_csv("C:\learn\ML\ML00caseStudy\week01Intro\song_data.csv")


# In[4]:
# getting no of rows and columns
data.num_rows(), data.num_columns() , len(data)


# In[5]:
# saving the dataset
data.save('songs')


# In[6]:
# loading the dataset
sdata = gl.load_sframe('songs')


# In[7]:
# head 5 rows
sdata.head(5)


# In[8]:
# likewise tail 4 rows
sdata.tail(4)


# In[9]:
# checking the column names and data type
sdata.column_types(), sdata.column_names()


# ### Modify the data

# In[10]:
# creating a new columns
sdata['my_rating'] = 0
sdata


# In[11]:
# generating a new colmns called song_age respective to my age

my_age = 1989

sdata['song_age']=sdata['year']-my_age
sdata


# In[12]:
# generating another column title length
sdata['title length'] = sdata['title'].apply(lambda x : len(x.split()))
sdata


# In[13]:
# adding one more column
sdata.add_column(sdata.select_column('year').apply(lambda x: x - my_age),name='how_old_was_i')
sdata


# In[14]:
# can add multiple columns at a time
sdata[['col1','col2']] = [sdata['title length'],sdata['my_rating']]
sdata


# In[15]:
#adding some more
#sdata[['col3','col4']] = [[3,4]]  this is not allowed, assigned values should be SArrray
#sdata


# In[16]:
# deleting column
del sdata['song_age']


# In[17]:
# rename columns
sdata.rename({'title length':'title_length'})
sdata


# In[18]:
# swap columns location
sdata.swap_columns('year','title_length')
sdata


# In[19]:
# change the column data type
sdata.column_names(),sdata.column_types()


# In[20]:

sdata['my_rating'] = sdata['my_rating'].astype(float)
sdata.column_types()


# In[21]:
# swap column location
sdata.swap_columns('title_length','year')


# In[22]:
# let's try to calculate how many 'a' are there in 'title', 'release' & 'artist_name' column
sdata.add_column(sdata['title', 'release' , 'artist_name'].apply(lambda row:sum(word.count('a') for word in row.values())),'no_of_a')
sdata.head(3)


# ## Checking Missing values

# In[23]:
# checking the missing year
year_count = sdata.groupby('year', gl.aggregate.COUNT)
year_count.head(6)


# In[24]:
# no of unique years
print "No of unique years :", str(len(year_count))


# In[25]:

print "no of invalid year count "
year_count.topk('year', reverse=True, k=1)


# In[26]:
# year 0 is invalid value to better convert it to None
sdata['year'] = sdata['year'].apply(lambda x : None if x==0 else x)
sdata.head(5)


# In[27]:
# now I have to fix the value of 'How_old_Was_i' column as well
# I can do this two ways, one - substract - year - born_year 
# or I can use the same apply function with NONE if x < 0
sdata['how_old_was_i'] = sdata['year'].apply(lambda x : None if x is None else x-my_age)
sdata.head(5)


# In[28]:
# check the no of songs which have valid years
print len(sdata[sdata['year']>0])


# In[29]:

tmp = sdata['year']>0
print tmp


# In[30]:
# Look at lots of descriptive statistics of title_length
print "mean: " + str(sdata['title_length'].mean())
print "std: " + str(sdata['title_length'].std())
print "var: " + str(sdata['title_length'].var())
print "min: " + str(sdata['title_length'].min())
print "max: " + str(sdata['title_length'].max())
print "sum: " + str(sdata['title_length'].sum())
print "number of non-zero entries: " + str(sdata['title_length'].nnz())


# In[31]:

approx_sketch = sdata['title_length'].sketch_summary()
print approx_sketch


# In[32]:
# lets check which songs are having largest and smallest length
top_title_length = sdata.topk('title_length')
print top_title_length


# In[33]:
# what about lowest
lowest_title_lenght = sdata.topk('title_length', reverse=True)
print lowest_title_lenght


# In[34]:
# lowest 15
# what about lowest
lowest_title_lenght = sdata.topk('title_length', reverse=True, k =15)
print lowest_title_lenght


# In[35]:

before_i_was_born = sdata['how_old_was_i'] < 0
before_i_was_born.all(), before_i_was_born.any()


# In[36]:
# get total songs in an album and display the top album by no
sdata.groupby(['artist_name','release'], {'no_of_songs_in_album':gl.aggregate.COUNT} ).topk('no_of_songs_in_album')


# In[37]:
# this will download the 118 MB file
#usage_data = gl.SFrame.read_csv("https://static.turi.com/datasets/millionsong/10000.txt", header=False, delimiter='\t', column_type_hints={'X3':int})
#usage_data.rename({'X1':'user_id', 'X2':'song_id', 'X3':'listen_count'})


# In[38]:
# Read the data
usage_data = gl.SFrame.read_csv("C:/learn/ML/ML00caseStudy/week01Intro/10000.txt", header=False, delimiter="\t", column_type_hints={'X3':int})
usage_data.rename({'X1':'user_id', 'X2':'song_id', 'X3':'listen_count'})


# In[39]:
# saving this data frame
usage_data.save('usage_data')
# loading  the data
usage_data = gl.load_sframe('usage_data')


# In[40]:
# find out the unique users
print len(usage_data['user_id'].unique())


# In[41]:
# let's create two datasets which we can join
ds1 = sdata[((sdata['artist_name'] == 'Relient K')
                           | (sdata['artist_name'] == 'Streetlight Manifesto'))
                          & (sdata['how_old_was_i'] >= 14) & (sdata['how_old_was_i'] <= 18)]
ds1


# In[42]:
# Let's join ds1 with the 10000.txt dataset usage_data
dsjoin = ds1.join(usage_data, 'song_id')
dsjoin


# In[43]:
# total row in ds1 and dsjoin datasets
len(ds1), len(dsjoin)


# In[44]:

len(ds1['song_id'].unique()), len(dsjoin['song_id'].unique()), len(ds1['song_id']), len(dsjoin['song_id'])


# In[45]:
# find out most popular songs when I was between 14 n 18
most_popular = dsjoin.groupby(['song_id'], {'total_listen_count':gl.aggregate.SUM('listen_count'), 
                                             'num_unique_users':gl.aggregate.COUNT('user_id')})
most_popular


# In[46]:
# to get artist name we have to join this data
most_popular.join(sdata, 'song_id').topk('total_listen_count',k=20)


# In[47]:
# let's append a row with max liste count and check whether it comes in above result or not
me = gl.SFrame({'user_id':['evan'],'song_id':['SOSFAVU12A6D4FDC6A'],'listen_count':[4000]})
# adding this data to usage data
usage_data = usage_data.append(me)


# In[48]:
# repeating the above join n group by statement
dsjoin = ds1.join(usage_data, 'song_id')
most_popular = dsjoin.groupby(['song_id'], {'total_listen_count':gl.aggregate.SUM('listen_count'), 
                                             'num_unique_users':gl.aggregate.COUNT('user_id')})
most_popular.join(sdata, 'song_id').topk('total_listen_count',k=20)


# ### Splitting and Sampling
# Lets check, how we can randomly split the data for test

# In[49]:
# Randomly split data rows into two subsets
first_set, second_set = sdata.random_split(0.8, seed = 1)
first_set.num_rows(), second_set.num_rows()


# If you want to split on a predicate though, you'll have to do that manually.
# In[50]:
songs_before = sdata[sdata['how_old_was_i'] < 0]
songs_after = sdata[sdata['how_old_was_i'] >= 0]
songs_before.num_rows(), songs_after.num_rows()


# In[51]:
# generating sample data
sample = sdata.sample(0.4)
sample.num_rows()


# ### SArray
# In[52]:

arr = gl.SArray([1,2,3])
arr


# In[53]:

arr2 = 2*arr
arr2


# In[54]:
# add
arr + arr2


# In[55]:
# multiply
arr * arr2


# In[56]:
# divide
arr2 / arr


# In[57]:
# iterating with SFrame
for i in sdata:
    if i['title_length'] >= 45:
        print "Whoa that's long!"


# ### Using apply function on SFrame
# In[58]:

sdata['title_artist_length'] = sdata['title','artist_name'].apply(lambda row: sum([len(col) for col in row.values()]))
sdata


# ### Saving Our Work

# In[59]:
# save as csv
sdata.save('sdata_new.csv', format='csv')


# In[60]:
sdata.save('sdata_new')






