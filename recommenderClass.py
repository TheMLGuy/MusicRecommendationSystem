import numpy as np
import pandas as pd

class popularityRecommender():
    def __init__(self):
        self.train_data=None
        self.user_id=None
        self.item_id=None
        self.popularity_recommendations=None
    #this function generates a popularity based recommender model    
    def create(self,train_data,user_id,item_id):
        self.train_data=train_data
        self.user_id=user_id
        self.item_id=item_id
        
        #item id corresponds to each unique songs, obj is to obtain 
        #count of user ids for each unique song
        grouped_train_data=train_data.groupby([self.item_id]).agg({self.user_id:'count'}).reset_index()
        print("printing 2 rows of training data")
        
        grouped_train_data.rename(columns={'user_id':'score'}, inplace=True)
        print(grouped_train_data.head(2))
        
        #sort songs based on column score
        grouped_sorted_data=grouped_train_data.sort_values(['score'],ascending=0)
        
        #generate a rank based on score
        grouped_sorted_data['rank']=grouped_sorted_data['score'].rank(ascending=0,method='first')
        
        #get top 10 recommendations
        self.popularity_recommendations=grouped_sorted_data.head(10)
    
    #popularity based recommender models just return the top preferred
    #suggestions irrespective of users likes
    def recommend(self,user_id):
        user_recommendations=self.popularity_recommendations
        user_recommendations['user_id']=user_id
        print("generating recommendations for user_id {}".format(user_id))
        cols=user_recommendations.columns.tolist()
        cols=cols[-1:]+cols[:-1]
        user_recommendations=user_recommendations[cols]
        return user_recommendations

class item_similarity_recommender():
    def __init__():
        self.train_data=None
        self.user_id=None
        self.item_id=None
        self.cooccurence_matrix=None
        self.songs_dict=None
        self.revs_songs_dict=None
        self.item_similarity_recommendations=None
    
    def get_unique_songs_for_user(self,user):
        user_data=self.train_data[self.train_data[self.user_id]==user]
        user_songs=list(user_data[self.item_id].unique())
        return user_songs
    
    def get_unique_users_for_song(self,item):
        item_data=self.train_data[self.train_data[self.item_id]==item]
        unique_users=list(item_data[self.user_id].unique())
        return unique_users
    
    def get_all_items_train_data(self):
        all_items=list(self.train_data[self.item_id].unique())
        return all_items
    
    def construct_coccurence_matrix(self,user_songs,all_songs):
        # get list of users for all songs
        user_songs_list=[]
        for i in range(len(user_songs)):
            user_songs_list.append(get_unique_songs_for_user(user_songs[i]))
        
        coocurence_matrix=np.matrix(np.zeros(shape=(len(user_songs), len(all_songs))),float)
        
        for i in range(len(all_songs)):
            songs_i_data=self.train_data[self.train_data[self.item_id]==all_songs[i]]
            users_i=set(songs_i_data[self.user_id].unique())
            
            for j in range(len(user_songs)):
                users_j=user_songs_list[j]
                users_intersection=users_i.intersection(users_j)
                
                if len(users_intersection)!=0:
                    users_union=users_i.union(users_j)
                    
                    coocurence_matrix[j,i]=float(len(users_intersection))/float(len(users_union))
                else:
                    coocurence_matrix[j,i]=0
                    
        return cooccurence_matrix
    
    def generate_top_recommendations(self,user,coocurence_matrix,all_songs,user_songs):
        print("non zero value in coocurence matrix {}".format(np.count_nonzero(coocurence_matrix)))
        
        user_sim_scores=cooccurence_matrix.sum(axis=0)/float(cooccurence_matrix.shape[0])
        user_sim_scores=np.array(user_sim_scores)[0].tolist()
        
        sort_index=sorted(((e,i) for i,e in enumerate(list(user_sim_scores))),reverse=True)
                          
        
        columns=['user_id','song','score','rank']
        df=pandas.DataFrame(columns=columns)
        
        rank=1                
        for i in range(len(sort_index)):
            if ~np.isnan(sort_index[i][0]) and all_songs[sort_index[i][1]] not in user_songs and rank<=10:
                df.loc[len(df)]=[user,all_songs[sort_index[i][1]], sort_index[i][0], rank]
                rank+=1
                          
        if df.shape[0]==0:
            print("current user has no songs for training item similarity based recommendation model")   
            return -1
        else:
            return df
                          
    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

    #Use the item similarity based recommender system model to
    #make recommendations
    def recommend(self, user):
        
        ########################################
        #A. Get all unique songs for this user
        ########################################
        user_songs = self.get_user_items(user)    
            
        print("No. of unique songs for the user: %d" % len(user_songs))
        
        ######################################################
        #B. Get all unique items (songs) in the training data
        ######################################################
        all_songs = self.get_all_items_train_data()
        
        print("no. of unique songs in the training set: %d" % len(all_songs))
         
        ###############################################
        #C. Construct item cooccurence matrix of size 
        #len(user_songs) X len(songs)
        ###############################################
        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs, all_songs)
        
        #######################################################
        #D. Use the cooccurence matrix to make recommendations
        #######################################################
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_songs, user_songs)
                
        return df_recommendations
    
    #Get similar items to given items
    def get_similar_items(self, item_list):
        
        user_songs = item_list
        
        ######################################################
        #B. Get all unique items (songs) in the training data
        ######################################################
        all_songs = self.get_all_items_train_data()
        
        print("no. of unique songs in the training set: %d" % len(all_songs))
         
        ###############################################
        #C. Construct item cooccurence matrix of size 
        #len(user_songs) X len(songs)
        ###############################################
        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs, all_songs)
        
        #######################################################
        #D. Use the cooccurence matrix to make recommendations
        #######################################################
        user = ""
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_songs, user_songs)
         
        return df_recommendations                    
    