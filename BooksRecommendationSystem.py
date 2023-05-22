import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from scipy.sparse import csr_matrix
from pandas.api.types import is_numeric_dtype
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import warnings
warnings.filterwarnings("ignore")




def RecommendationSystem(bookInput, numRecommendations):
    # loading_data()
    books = pd.read_csv(r"Datasets/Books.csv", delimiter=';', error_bad_lines=False, encoding='ISO-8859-1', warn_bad_lines=False)
    users = pd.read_csv(r"Datasets/Users.csv", delimiter=';', error_bad_lines=False, encoding='ISO-8859-1', warn_bad_lines=False)
    ratings = pd.read_csv(r"Datasets/Book-Ratings.csv", delimiter=';', error_bad_lines=False, encoding='ISO-8859-1', warn_bad_lines=False)

    print("Books Data:    ", books.shape)
    print("Users Data:    ", users.shape)
    print("Books-ratings: ", ratings.shape)

    #### books_preprocessing

    print("Columns: ", list(books.columns))
    print(books.head())

    ## Drop URL columns
    books.drop(['Image-URL-S', 'Image-URL-L'], axis=1, inplace=True)
    books.head()

    ## Checking for null values
    books.isnull().sum() 

    books.loc[books['Book-Author'].isnull(),:]
    books.loc[books['Publisher'].isnull(),:]

    books.at[187689 ,'Book-Author'] = 'Other'
    books.at[128890 ,'Publisher'] = 'Other'
    books.at[129037 ,'Publisher'] = 'Other' 

    ## Checking for column Year-of-publication
    books['Year-Of-Publication'].unique()

    pd.set_option('display.max_colwidth', -1)

    books.loc[books['Year-Of-Publication'] == 'DK Publishing Inc',:]
    books.loc[books['Year-Of-Publication'] == 'Gallimard',:]

    books.at[209538 ,'Publisher'] = 'DK Publishing Inc'
    books.at[209538 ,'Year-Of-Publication'] = 2000
    books.at[209538 ,'Book-Title'] = 'DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)'
    books.at[209538 ,'Book-Author'] = 'Michael Teitelbaum'

    books.at[221678 ,'Publisher'] = 'DK Publishing Inc'
    books.at[221678 ,'Year-Of-Publication'] = 2000
    books.at[209538 ,'Book-Title'] = 'DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)'
    books.at[209538 ,'Book-Author'] = 'James Buckley'

    books.at[220731 ,'Publisher'] = 'Gallimard'
    books.at[220731 ,'Year-Of-Publication'] = '2003'
    books.at[209538 ,'Book-Title'] = 'Peuple du ciel - Suivi de Les bergers '
    books.at[209538 ,'Book-Author'] = 'Jean-Marie Gustave Le ClÃ?Â©zio'

    ## Converting year of publication in Numbers
    books['Year-Of-Publication'] = books['Year-Of-Publication'].astype(int)

    print(sorted(list(books['Year-Of-Publication'].unique())))

    ## Replacing Invalid years with max year
    count = Counter(books['Year-Of-Publication'])
    [k for k, v in count.items() if v == max(count.values())]

    books.loc[books['Year-Of-Publication'] > 2021, 'Year-Of-Publication'] = 2002
    books.loc[books['Year-Of-Publication'] == 0, 'Year-Of-Publication'] = 2002

    ## Uppercasing all alphabets in ISBN
    books['ISBN'] = books['ISBN'].str.upper()

    ## Drop duplicate rows
    books.drop_duplicates(keep='last', inplace=True) 
    books.reset_index(drop = True, inplace = True)

    books.info()
    print(books.head())


    #### users_preprocessing

    print("Columns: ", list(users.columns))
    users.head()


    ## Checking null values
    print(users.isna().sum())  

    ## Check for all values present in Age column
    print(sorted(list(users['Age'].unique())))   

    required = users[users['Age'] <= 80]
    required = required[required['Age'] >= 10]      

    mean = round(required['Age'].mean())   
    users.loc[users['Age'] > 80, 'Age'] = mean    #outliers with age grater than 80 are substituted with mean 
    users.loc[users['Age'] < 10, 'Age'] = mean    #outliers with age less than 10 years are substitued with mean
    users['Age'] = users['Age'].fillna(mean)      #filling null values with mean
    users['Age'] = users['Age'].astype(int)       #changing Datatype to int

    list_ = users.Location.str.split(', ')

    city = []
    state = []
    country = []
    count_no_state = 0    
    count_no_country = 0

    for i in range(0,len(list_)):
        if list_[i][0] == ' ' or list_[i][0] == '' or list_[i][0]=='n/a' or list_[i][0] == ',':  #removing invalid entries too
            city.append('other')
        else:
            city.append(list_[i][0].lower())

        if(len(list_[i])<2):
            state.append('other')
            country.append('other')
            count_no_state += 1
            count_no_country += 1
        else:
            if list_[i][1] == ' ' or list_[i][1] == '' or list_[i][1]=='n/a' or list_[i][1] == ',':   #removing invalid entries 
                state.append('other')
                count_no_state += 1            
            else:
                state.append(list_[i][1].lower())
            
            if(len(list_[i])<3):
                country.append('other')
                count_no_country += 1
            else:
                if list_[i][2] == ''or list_[i][1] == ',' or list_[i][2] == ' ' or list_[i][2] == 'n/a':
                    country.append('other')
                    count_no_country += 1
                else:
                    country.append(list_[i][2].lower())
            
    users = users.drop('Location',axis=1)

    temp = []
    for ent in city:
        c = ent.split('/')            #handling cases where city/state entries from city list as state is already given 
        temp.append(c[0])

    df_city = pd.DataFrame(temp,columns=['City'])
    df_state = pd.DataFrame(state,columns=['State'])
    df_country = pd.DataFrame(country,columns=['Country'])

    users = pd.concat([users, df_city], axis=1)
    users = pd.concat([users, df_state], axis=1)
    users = pd.concat([users, df_country], axis=1)

    print(count_no_country)   #printing the number of countries didnt have any values 
    print(count_no_state)     #printing the states which didnt have any values

    ## Drop duplicate rows
    users.drop_duplicates(keep='last', inplace=True)
    users.reset_index(drop=True, inplace=True)

    users.info()
    print(users.head())

    ### books_rating preprocessing
    print("Columns: ", list(ratings.columns))
    print(ratings.head())

    ## Checking for null values
    ratings.isnull().sum() 

    ## checking all ratings number or not
    print(is_numeric_dtype(ratings['Book-Rating']))

    ## checking User-ID contains only number or not
    print(is_numeric_dtype(ratings['User-ID']))

    ## checking ISBN
    flag = 0
    k =[]
    reg = "[^A-Za-z0-9]"

    for x in ratings['ISBN']:
        z = re.search(reg,x)    
        if z:
            flag = 1

    if flag == 1:
        print("False")
    else:
        print("True")


    ## removing extra characters from ISBN (from ratings dataset) existing in books dataset
    bookISBN = books['ISBN'].tolist() 
    reg = "[^A-Za-z0-9]" 
    for index, row_Value in ratings.iterrows():
        z = re.search(reg, row_Value['ISBN'])    
        if z:
            f = re.sub(reg,"",row_Value['ISBN'])
            if f in bookISBN:
                ratings.at[index , 'ISBN'] = f


    ## Uppercasing all alphabets in ISBN
    ratings['ISBN'] = ratings['ISBN'].str.upper()

    ## Drop duplicate rows
    ratings.drop_duplicates(keep='last', inplace=True)
    ratings.reset_index(drop=True, inplace=True)

    ratings.info()

    print(ratings.head())

    ### Merging of Users, Books and Rating Tables in One
    dataset = pd.merge(books, ratings, on='ISBN', how='inner')
    dataset = pd.merge(dataset, users, on='User-ID', how='inner')
    dataset.info()

    ## Explicit Ratings Dataset
    dataset1 = dataset[dataset['Book-Rating'] != 0]
    dataset1 = dataset1.reset_index(drop = True)
    dataset1.shape

    ## Implicit Ratings Dataset
    dataset2 = dataset[dataset['Book-Rating'] == 0]
    dataset2 = dataset2.reset_index(drop = True)
    dataset2.shape

    print(dataset1.head())

    # bookName = input("Enter a book name: ")
    # number = int(input("Enter number of books to recommend: "))
    bookName = bookInput
    number = int(numRecommendations)

    #Harry Potter and the Sorcerer's Stone (Harry Potter (Paperback))

    ### Collaborative Filtering (User-Item Filtering)
    df = pd.DataFrame(dataset1['Book-Title'].value_counts())
    df['Total-Ratings'] = df['Book-Title']
    df['Book-Title'] = df.index
    df.reset_index(level=0, inplace=True)
    df = df.drop('index',axis=1)

    df = dataset1.merge(df, left_on = 'Book-Title', right_on = 'Book-Title', how = 'left')
    df = df.drop(['Year-Of-Publication','Publisher','Age','City','State','Country'], axis=1)

    popularity_threshold = 50
    popular_book = df[df['Total-Ratings'] >= popularity_threshold]
    popular_book = popular_book.reset_index(drop = True)

    testdf = pd.DataFrame()
    testdf['ISBN'] = popular_book['ISBN']
    testdf['Book-Rating'] = popular_book['Book-Rating']
    testdf['User-ID'] = popular_book['User-ID']
    testdf = testdf[['User-ID','Book-Rating']].groupby(testdf['ISBN'])

    listOfDictonaries=[]
    indexMap = {}
    reverseIndexMap = {}
    ptr=0

    for groupKey in testdf.groups.keys():
        tempDict={}
        groupDF = testdf.get_group(groupKey)
        for i in range(0,len(groupDF)):
            tempDict[groupDF.iloc[i,0]] = groupDF.iloc[i,1]
        indexMap[ptr]=groupKey
        reverseIndexMap[groupKey] = ptr
        ptr=ptr+1
        listOfDictonaries.append(tempDict)

    dictVectorizer = DictVectorizer(sparse=True)
    vector = dictVectorizer.fit_transform(listOfDictonaries)
    pairwiseSimilarity = cosine_similarity(vector)

    def printBookDetails(bookID):
        print(dataset1[dataset1['ISBN']==bookID]['Book-Title'].values[0])
        """
        print("Title:", dataset1[dataset1['ISBN']==bookID]['Book-Title'].values[0])
        print("Author:",dataset1[dataset['ISBN']==bookID]['Book-Author'].values[0])
        #print("Printing Book-ID:",bookID)
        print("\n")
        """

    def getTopRecommandations(bookID):
        collaborative = []
        row = reverseIndexMap[bookID]
        print("Input Book:")
        printBookDetails(bookID)
        
        print("\nRECOMMENDATIONS:\n")
        
        mn = 0
        similar = []
        for i in np.argsort(pairwiseSimilarity[row])[:-2][::-1]:
            if dataset1[dataset1['ISBN']==indexMap[i]]['Book-Title'].values[0] not in similar:
                    if mn>=number:
                        break
                    mn+=1
                    similar.append(dataset1[dataset1['ISBN']==indexMap[i]]['Book-Title'].values[0])
                    printBookDetails(indexMap[i])
                    collaborative.append((dataset1[dataset1['ISBN']==indexMap[i]]['Book-Title'].values[0], dataset1[dataset1['ISBN']==indexMap[i]]['Image-URL-M'].values[0]))
        return collaborative
    
    k = list(dataset1['Book-Title'])
    m = list(dataset1['ISBN'])

    collaborative = getTopRecommandations(m[k.index(bookName)])
    # print(collaborative)

    ### Content Based
    popularity_threshold = 80
    popular_book = df[df['Total-Ratings'] >= popularity_threshold]
    popular_book = popular_book.reset_index(drop = True)
    popular_book.shape

    tf = TfidfVectorizer(ngram_range=(1, 2), min_df = 1, stop_words='english')
    tfidf_matrix = tf.fit_transform(popular_book['Book-Title'])
    tfidf_matrix.shape

    normalized_df = tfidf_matrix.astype(np.float32)
    cosine_similarities = cosine_similarity(normalized_df, normalized_df)
    cosine_similarities.shape

    print("Recommended Books:\n")
    isbn = books.loc[books['Book-Title'] == bookName].reset_index(drop = True).iloc[0]['ISBN']
    content = []

    idx = popular_book.index[popular_book['ISBN'] == isbn].tolist()[0]
    similar_indices = cosine_similarities[idx].argsort()[::-1]
    similar_items = []
    for i in similar_indices:
        if popular_book['Book-Title'][i] != bookName and popular_book['Book-Title'][i] not in similar_items and len(similar_items) < number:
            similar_items.append(popular_book['Book-Title'][i])
            content.append((popular_book['Book-Title'][i], popular_book['Image-URL-M'][i]))

    for book in similar_items:
        print(book)
    
    final_lst = list()
    final_lst.append(collaborative) ### Collaborative Filtering
    final_lst.append(content) ### Content Based FIltering

    return final_lst
# if __name__ == "__main__":
# lst = []
# lst = RecommendationSystem()
# print(lst)
## [['Harry Potter and the Prisoner of Azkaban (Book 3)', 'Harry Potter and the Goblet of Fire (Book 4)', 'Harry Potter and the Order of the Phoenix (Book 5)','Harry Potter and the Chamber of Secrets (Book 2)', 'Fried Green Tomatoes at the Whistle Stop Cafe'], ['Harry Potter and the Sorcerer's Stone (Book 1)', 'Harry Potter and the Goblet of Fire (Book 4)', 'Harry Potter and the Chamber of Secrets (Book 2)', 'Harry Potter and the Prisoner of Azkaban (Book 3)', 'Harry Potter and the Order of the Phoenix (Book 5)']]