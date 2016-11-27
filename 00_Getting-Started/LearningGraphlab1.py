
# importing graphlab 
import graphlab as gl

# creating SFrame object by reading a csv file
sf = gl.SFrame('C:\learn\ML\ML00caseStudy\people.csv')

# view content of sf variable
sf

# printing head 5 line, first 5 lines
sf.head(5)

# likewise tail
sf.tail(3)

