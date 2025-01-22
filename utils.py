import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline


# EXAMPLE CUSTOM PIPELINES
#
# cust_pipeline = Pipeline([
#     ( 'GenerateMissingIndicator: EDUCATION', GenerateMissingIndicator(['EDUCATION']) ),
#     ( 'GroupRareLevels: EDUCATION, MARRIAGE', GroupRareLevels(feat_lst=['EDUCATION', 'MARRIAGE']) ),
#     ( 'OutlierRemovalIQR: AGE, LIMIT_BAL', OutlierRemovalIQR(['AGE', 'LIMIT_BAL']) )
# ])
#
# df_new = cust_pipeline.fit_transform(df)




# Group rare levels of categorical variables
class GroupRareLevels(BaseEstimator, TransformerMixin):
    
    def __init__(self, p_thresh=0.01, feat_lst=None):
        print('---- GroupRareLevels init method ----')
        
        self._feat_lst = {}
        self._p_thresh = p_thresh 
        
        # build the hash table with the lower and upper limits for each feature
        if feat_lst == None:
            feat_lst = []
        else:
            for feat in feat_lst:
                self._feat_lst[feat] = []        
        
        print(self._p_thresh)
        
    def fit(self, x, y=None):
        print('---- GroupRareLevels fit method ----')
        
        # figure out what are the rare levels for each feature
        for feat in self._feat_lst:
            tmp = dict( x[feat].value_counts(normalize=True) )
            print(feat)
            print(tmp)
            self._feat_lst[feat] = [k for k in tmp if tmp[k] <= self._p_thresh]
            
        print('Features :', self._feat_lst)
        
        return self
        
        
    def transform(self, x):
        print('---- GroupRareLevels transform method ----')
        for feat in self._feat_lst:
            ind = x[feat].isin(self._feat_lst[feat])
            x.loc[ind, feat] = 'other'
        
        return x
    
    
# Generate a missing indicator - useful for numeric attributes
class GenerateMissingIndicator(BaseEstimator, TransformerMixin):
    
    def __init__(self, feat_lst=None):
        print('---- GenerateMissingIndicator init method ----')
                
        if feat_lst == None:
            feat_lst = []
        
        self._feat_lst = feat_lst
        
        print('Features :', self._feat_lst)
        
        
    def fit(self, x, y=None):
        print('---- GenerateMissingIndicator fit method ----')
        # here we do not do anything
        return self
        
        
    def transform(self, x):
        print('---- GenerateMissingIndicator transform method ----')
        for feat in self._feat_lst:
            x[feat+'_missing'] = 0
            ind = x[feat].isna()
            x.loc[ind, feat+'_missing'] = 1
        
        return x
    

# 1D outlier removal using IQR
class OutlierRemovalIQR(BaseEstimator, TransformerMixin):
    
    def __init__(self, feat_lst=None):
        print('---- OutlierRemovalIQR init method ----')
        
        self._feat_lst = {}
        
        # build the hash table with the lower and upper limits for each feature
        if feat_lst == None:
            feat_lst = []
        else:
            for feat in feat_lst:
                self._feat_lst[feat] = []
        
        print('Features :', self._feat_lst)
        
        
    def fit(self, x, y=None):
        print('---- OutlierRemovalIQR fit method ----')
        
        # get the lower and upper limits
        for feat in self._feat_lst:
            percentile25 = x[feat].quantile(0.25)
            percentile75 = x[feat].quantile(0.75)
            
            iqr = percentile75 - percentile25
            upper_limit = percentile75 + 1.5 * iqr
            lower_limit = percentile25 - 1.5 * iqr
            
            self._feat_lst[feat] = [lower_limit, upper_limit]
            
        print('Features :', self._feat_lst)
        
        return self
        
        
    def transform(self, x):
        print('---- OutlierRemovalIQR transform method ----')
        for feat in self._feat_lst:
            
            ind_1 = x[ x[feat] < self._feat_lst[feat][0] ].index
            ind_2 = x[ x[feat] > self._feat_lst[feat][1] ].index
            
            x.drop(ind_1, inplace=True)
            x.drop(ind_2, inplace=True)
        
        return x
    
    
# CountVectorizerTransformer
class CountVectorizerTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, feat_lst=None, drop_feat=False):
        print('---- CountVectorizerTransformer init method ----')
        
        self._feat_lst = {}
        self._drop_feat = drop_feat
        
        # build the hash table with the lower and upper limits for each feature
        if feat_lst == None:
            feat_lst = []
        else:
            for feat in feat_lst:
                self._feat_lst[feat] = []        
        
    def fit(self, x, y=None):
        print('---- CountVectorizerTransformer fit method ----')
        
        # train the transformer for each feature
        for feat in self._feat_lst:
            
            countvectorizer = CountVectorizer(analyzer= 'word', stop_words='english')
            countvectorizer.fit(x[feat])
            
            self._feat_lst[feat] = countvectorizer
            
        print('Features :', self._feat_lst)
        
        return self
        
        
    def transform(self, x):
        print('---- CountVectorizerTransformer transform method ----')
        for feat in self._feat_lst:
            
            countvectorizer = self._feat_lst[feat]
            count_wm = countvectorizer.transform(x[feat])
            count_tokens = ['cv_'+feat+'_'+tkn for tkn in countvectorizer.get_feature_names()]
            
            df_countvect = pd.DataFrame(data = count_wm.toarray(), index = x.index, columns = count_tokens)
            #print(df_countvect)
            
            if self._drop_feat:
                x.drop([feat], axis=1, inplace=True)
        
            x = x.join(df_countvect, how='inner')
        
        return x
    
    
    
# TfidfVectorizerTransformer
class TfidfVectorizerTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, feat_lst=None, drop_feat=False):
        print('---- TfidfVectorizerTransformer init method ----')
        
        self._feat_lst = {}
        self._drop_feat = drop_feat
        
        # build the hash table with the lower and upper limits for each feature
        if feat_lst == None:
            feat_lst = []
        else:
            for feat in feat_lst:
                self._feat_lst[feat] = []        
        
    def fit(self, x, y=None):
        print('---- TfidfVectorizerTransformer fit method ----')
        
        # train the transformer for each feature
        for feat in self._feat_lst:
            
            tfidfvectorizer = TfidfVectorizer(analyzer='word',stop_words= 'english')
            tfidfvectorizer.fit(x[feat])
            
            self._feat_lst[feat] = tfidfvectorizer
            
        print('Features :', self._feat_lst)
        
        return self
        
        
    def transform(self, x):
        print('---- TfidfVectorizerTransformer transform method ----')
        for feat in self._feat_lst:
            
            tfidfvectorizer = self._feat_lst[feat]
            tfidf_wm = tfidfvectorizer.transform(x[feat])
            tfidf_tokens = ['tfidf_'+feat+'_'+tkn for tkn in tfidfvectorizer.get_feature_names()]
            
            df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(), index = x.index, columns = tfidf_tokens)
            #print(df_tfidfvect)
            
            if self._drop_feat:
                x.drop([feat], axis=1, inplace=True)
        
            x = x.join(df_tfidfvect, how='inner')
        
        return x
    

    
# SplitStrTransformer
class SplitStrTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, feat_lst=None, pattern=' ', n=-1, drop_feat=False):
        print('---- SplitStrTransformer init method ----')
        
        self._feat_lst = {}
        self._pattern = pattern
        self._n = n
        self._drop_feat = drop_feat
        
        # build the hash table with the lower and upper limits for each feature
        if feat_lst == None:
            feat_lst = []
        else:
            for feat in feat_lst:
                self._feat_lst[feat] = []        
        
    def fit(self, x, y=None):
        print('---- SplitStrTransformer fit method ----')
        
        return self
        
        
    def transform(self, x):
        print('---- SplitStrTransformer transform method ----')
        for feat in self._feat_lst:
            
            df_tmp = x[feat].str.split(self._pattern, n=self._n, expand=True)
            new_col = ['text_split_'+str(c) for c in df_tmp.columns]
            df_tmp.rename(dict(zip(df_tmp.columns,new_col)), axis=1, inplace=True)
            #print(df_tmp)
            
            if self._drop_feat:
                x.drop([feat], axis=1, inplace=True)
        
            x = x.join(df_tmp, how='inner')
        
        return x
    
    

# LogTransformer
class LogTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, feat_lst=None):
        print('---- LogTransformer init method ----')
        
        if feat_lst == None:
            feat_lst = []
        
        self._feat_lst = feat_lst     
        
    def fit(self, x, y=None):
        print('---- LogTransformer fit method ----')
        
        return self
        
        
    def transform(self, x):
        print('---- SplitStrTransformer transform method ----')
        for feat in self._feat_lst:
            
            x.loc[:, feat] = np.log1p( x[feat] )
        
        return x