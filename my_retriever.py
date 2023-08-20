import math

class Retrieve:

    # Create new Retrieve object storing index and term weighting
    # scheme. (You can extend this method, as required.)
    def __init__(self, index, term_weighting):
        self.index = index
        self.term_weighting = term_weighting
        self.num_docs = self.compute_number_of_documents()
        self.df = self.get_df()
        
    def compute_number_of_documents(self):
        self.doc_ids = set() 
        for term in self.index:
            self.doc_ids.update(self.index[term])
        return len(self.doc_ids)

    #get document frequency
    def get_df(self):
        df={}
        for term in self.index:
            df[term] = len(self.index[term])
        return df

    #functions for documents term weighting
    def tf_vector(self, tf, term):
        for docId in self.index[term]:
            if docId in tf:
                tf[docId][term] = self.index[term][docId]
            else:
                tf[docId]={}
                tf[docId][term] = self.index[term][docId]
        return tf
    
    def bi_vector(self, bi, term):
        for docId in self.index[term]:
            if docId in bi:
                bi[docId][term] = 1
            else:
                bi[docId]={}
                bi[docId][term] = 1
        return bi

    def tfidf_vector(self, tfidf, tf, docId):
        tfidf[docId]={}
        for term in tf[docId]:
            idf = math.log(self.num_docs/self.df[term])
            tfidf[docId][term] = tf[docId][term]*idf
        return tfidf
        

    # functions for query term weighting
    def tf_query(self, query, tfQ):
        for term in query:
            if term in tfQ:
                tfQ[term] +=1
            else:
                tfQ[term] = 1
    
    def bi_query(self, query, biQ):
        for term in query:
            biQ[term] = 1

    def tfidf_query(self, tfQ, tfidfQ):
        for term in tfQ:
            if term in self.df:
                idf = math.log(self.num_docs/self.df[term])
            else:
                idf = 0
            tfidfQ[term] = tfQ[term]*idf
    
    def term_weighting_query(self, query, twQ):
        if self.term_weighting == 'tf':
            self.tf_query(query, twQ)
        elif self.term_weighting == 'tfidf':
            tfQ = {}
            self.tf_query(query, tfQ)
            self.tfidf_query(tfQ, twQ)
        else:
            self.bi_query(query, twQ)
    
    def similarity(self, vecQ, vecD):
        qd = 0
        d2 = 0
        for term in vecQ:
            if term in vecD:
                qd += vecQ[term]*vecD[term]
        for term in vecD:
            d2 += vecD[term]*vecD[term]
        d2 = math.sqrt(d2)
        sim = qd/d2
        return sim



    # Method performing retrieval for a single query (which is 
    # represented as a list of preprocessed terms). Returns list 
    # of doc ids for relevant docs (in rank order).
    def for_query(self, query):

        #query term weighting
        twQ={}
        self.term_weighting_query(query, twQ)
        
        #documents term weighting
        tw={}
        if self.term_weighting == 'tf':
            tf={}
            for term in self.index:
                tw = self.tf_vector(tf, term)
        elif self.term_weighting == 'tfidf':
            tf={}
            tfidf={}
            for term in self.index:
                tf = self.tf_vector(tf, term)
            for id in tf:
                tw = self.tfidf_vector(tfidf, tf, id)
        else:
            bi={}
            for term in self.index:
                tw = self.bi_vector(bi ,term)

        #get similarity
        simDic={}
        for doc in tw:
            sim = self.similarity(twQ, tw[doc])
            simDic[doc] = sim
            
        #sort documents
        result = sorted(simDic, key=simDic.get, reverse=True)[:10]
        return result


