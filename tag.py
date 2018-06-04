import pandas as pd
from pymystem3 import Mystem
import numpy as np
from gensim.models.phrases import Phraser, Phrases
from gensim.models import TfidfModel, Word2Vec, FastText
from gensim import corpora
import nltk
import matplotlib.pyplot as plt
from collections import namedtuple, Counter
import re
import glob
import os
from wordcloud import (WordCloud, get_single_color_func)
from PIL import Image
import matplotlib.pyplot as plt
from gensim.corpora import Dictionary

from time import time as t
from sklearn.externals import joblib
from sklearn.manifold import MDS, locally_linear_embedding, Isomap
from sklearn.decomposition import PCA

import plotly.offline as py

import plotly.figure_factory as ff

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go

init_notebook_mode(connected=True)

import artm

class TagData:
	__stop_word = ["в",'!','?' "без", "до", "из", "к", "на", "по", "о", "от", "перед",
	             "при", "через", "за", "над", "об", "под", "про", "для", "вблизи", 
	             "вглубь", "вдоль", "возле", "около", "вокруг", "впереди", "после", 
	             "посредством", "в","роли", "путём", "насчёт", "по","поводу", "ввиду", "по", "случаю", 
	             "в", "течение", "благодаря", "несмотря", "на", "спустя", "с", "из-под", 
	             "из-за", "я", "быть", "с", "это", "все", "то", "по-над", "отличие", "от", 
	             "в","связи", "как", "словно", "так","как","т.к.", "тоже", "зато", "чтобы", 
	             "также", "потому","что", "и", "а", "что", "или", "но", "однако", "когда", "лишь", 
	             "едва", "где", "куда", "откуда", "столько", "настолько", "так", "такой","степени", "до",
	             "того", "такой",  "будто", "точно", "если",  "коли", "ежели", "несмотря", "хотя", "хоть", 
	             "пускай", "дабы", "-","же",":" "ли", "какой", "этот", "весь", "бы", "то", "у", 
	             "уже","который","там","оно","ничто","кто","вот","тот","ну","со" "даже","она","он", ' ']


	def get_stopwords(self):
		return self.__stop_word

	def set_stopwords(self, stop_word):
		self.__stop_word = stop_word


	__keys = pd.read_json('keys.json')

	def get_keys(self):
		return self.__keys


	__dict_tag = joblib.load('tag.model')

	def get_dict_tag(self):
		return self.__dict_tag



	__w2v = Word2Vec.load('w2v1.model')

	def get_w2v(self):
		return self.__w2v


	__bigram_transformer = Phraser.load('bi_trans.phr')

	def get_bgramm(self):
		return self.__bigram_transformer



	__trigram_trasformer = Phraser.load('tri_trans.phr')

	def get_tgramm(self):
		return self.__trigram_trasformer



	__fastText = FastText.load('fast.model')

	def get_fast(self):
		return self.__fastText


TagData = TagData()


class TagComment:

	__path = ''
	__df = None
	__trigram_corp = None
	__df_tag = None
	__df_attrharr = None
	__feature_array = None


	def __init__(self, path, csv=True):
		self.__path = path
		if csv:
			try:
				self.__df = pd.read_csv(path)
			except:
				print('неверный формат')
		else:
			try:
				self.__df = pd.read_excel(path)
			except:
				print('неверный формат')

	


	def fit(self):
		df = self.__df
		df.fillna('', inplace=True)
		df['comment'] = df.title + ' ' + df.content


		drop_comment = []
		alph = 'йцукенгшщзхъфывапролджэячсмитьбю'
		for i in range(len(df)):
		    delete = False
		    for char in df.comment.loc[i]:
		        if char in alph:
		            delete = False
		            break
		        else:
		            delete = True
		    if delete:
		        drop_comment.append(i)
		df.drop(drop_comment,axis=0,inplace=True)
		df.index = range(len(df))


		df['lemma'] = ''

		m = Mystem()

		idx = 0
		for comment in df.comment.as_matrix():
		    lemma = m.lemmatize(comment)[:]
		    df.lemma.loc[idx] = ''.join(lemma[1:len(lemma)- 1])
		    if idx%20000 == 0:
		        0
		        #print(idx)
		    idx +=1
		print('lemmatisation end')



		stop_word = TagData.get_stopwords()
		wpt = nltk.TweetTokenizer()
		tokenized_corpus_viz = [wpt.tokenize(re.sub('[/\nt,.qwtyuipadghjklzxcbn$&^=+"]', '', str(document))) for document in df['lemma'].str.lower()]
		print('tokenize end')
		tmp = []
		t0 = t()
		z = 0
		for j in tokenized_corpus_viz:
		    d = [doc not in stop_word for doc in j]
		    d = list(np.array(j)[d])
		    k = 0
		    dtemp = []
		    for i in range(len(d)):
		        #print(d[i])
		        if d[i] == 'не':
		            try:
		                dtemp.append(d[i] + '_' + d[i+1])
		                k =1
		            except:
		                0
		        else:
		            if k == 0:
		                dtemp.append(d[i])
		            else:
		                k = 0
		    d = dtemp
		    tmp.append(d)
		    if z%100000 ==0:
		        0
		        #print(z)
		    z+=1
		tokenized_corpus_viz = tmp
		t1 = t()
		print("%s: %.2g sec" % ('stop delete end: ', int(t1 - t0)))

		
		fastText = TagData.get_fast()
		new_tokenize =[]
		k = 0
		t0 = t()
		for rew in tokenized_corpus_viz:
		    new_rew = []
		    for word in rew:
		        try:
		            
		            if fastText.wv.vocab[word].index > 1400:
		                if fastText.wv.most_similar(word, restrict_vocab=1400, topn=1)[0][1] >= 0.52:
		                    new_rew.append(fastText.wv.most_similar(word, restrict_vocab=1400, topn=1)[0][0])
		            else:
		                new_rew.append(word)
		        except:
		            try:
		                if fastText.wv.most_similar(word, restrict_vocab=1400, topn=1)[0][1] >= 0.52:
		                    new_rew.append(fastText.wv.most_similar(word, restrict_vocab=1400, topn=1)[0][0])
		            except:
		                0
		    new_tokenize.append(new_rew)
		    if k%100000 == 0:
		        #print(k)
		        #print(t() - t0)
		        #t0 = t()
		        0
		    k+=1

		print('spellcheker end')
		bigram_transformer = TagData.get_bgramm()
		trigram_trasformer = TagData.get_tgramm()

		def tex_gen_trigram(tokenized_corpus):
		    trig = []
		    for text in tokenized_corpus:
		        trig.append(trigram_trasformer[bigram_transformer[[word for word in text]]])
		    return trig

		trigram_corp = tex_gen_trigram(new_tokenize)
		print('bi-tri gramm end')

		keys = TagData.get_keys()

		dict_tag = TagData.get_dict_tag()

		nnkk = keys.cl_id.drop_duplicates()
		df_attrharr = pd.DataFrame([0 for i in range(len(nnkk))], nnkk)

		kknn = keys.clust1.append(keys.clust2).drop_duplicates()
		df_tag = pd.DataFrame([0 for i in range(len(kknn))], kknn)

		for j in range(len(trigram_corp)):
			d = {}
			d1 = {}

			for i in trigram_corp[j]:
			    
			    try:
			        try:
			            d[dict_tag[i][0]] += 1

			        except:
			            d[dict_tag[i][0]] = 1


			        try:
			            d[dict_tag[i][1]] += 1

			        except:
			            d[dict_tag[i][1]] = 1


			        try:
			            d1[dict_tag[i][2]] += 1
			        except:
			            d1[dict_tag[i][2]] = 1
			    except:
			        0

			try:
			    d.pop('delete')
			except:
			    0

			try:
			    d1.pop('delete')
			except:
			    0

			for g in d.keys():
			    df_tag[j].loc[g] = d[g]
			for g in d1.keys():
			    df_attrharr[j].loc[g] = d1[g]

			df_tag[j+1] = 0
			df_attrharr[j+1] = 0 


		def average_word_vectors(words, model, vocabulary, num_features):
    
		    feature_vector = np.zeros((num_features,),dtype="float64")
		    nwords = 0.
		    
		    for word in words:
		        if word in vocabulary: 
		            nwords = nwords + 1.
		            feature_vector = np.add(feature_vector, model[word])
		    
		    if nwords:
		        feature_vector = np.divide(feature_vector, nwords)
		        
		    return feature_vector
	    
	   

		def averaged_word_vectorizer(corpus, model, num_features):
		    vocabulary = set(model.wv.index2word)
		    features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features)
		                    for tokenized_sentence in corpus]
		    return np.array(features)

		self.__trigram_corp = trigram_corp
		self.__df_tag = df_tag
		self.__df_attrharr = df_attrharr
		self.__feature_array = averaged_word_vectorizer(trigram_corp,TagData.get_w2v(),300)
		return df_tag, df_attrharr

	def get_trigram_corp(self):
		return self.__trigram_corp

	def get_tag(self):
		return self.__df_tag

	def get_attrharr(self):
		return self.__df_attrharr

	def get_data(self):
		return  self.__df

	def get_tagname(self):
		return self.__df_tag.index		

	def pipe(self, percent = 0.02, tag = None, tagOrattr = True):
		if tagOrattr:
			if tag == None:
				pipe = self.__df_tag.sum(axis=1)/self.__df_tag.sum(axis=1).sum()
			else:
				disign = self.__df_tag[list(self.__df_tag.columns[self.__df_tag.loc[tag] > 0])]
				pipe = disign.sum(axis=1)/disign.sum(axis=1).sum()
		else:
			pipe = self.__df_attrharr.sum(axis=1)/self.__df_attrharr.sum(axis=1).sum()

		labels = pipe[pipe>percent].index
		values = pipe[pipe>percent].values

		trace = go.Pie(labels=labels, values=values)

		py.iplot([trace], filename='basic_pie_chart')

	

	def scatter(self, manifolds = 'MDS'):
		if manifolds == 'MDS':
			mds = MDS(n_jobs = -1)
			data_scatter = mds.fit_transform(self.__feature_array)
		elif manifolds == 'PCA':
			pca = PCA(n_components=2)
			data_scatter = pca.fit_transform(self.__feature_array)
		elif manifolds == 'Iso':
			iso = Isomap(n_jobs = -1)
			data_scatter = iso.fit_transform(self.__feature_array)
		else:
			print('указан неверный параметр. Используется PCA')
			pca = PCA(n_components=2)
			data_scatter = pca.fit_transform(self.__feature_array)

		df_sc = pd.DataFrame(data_scatter)
		df = self.__df
		df.fillna('', inplace=True)
		df['comment'] = df.title + ' ' + df.content
		trace1 = go.Scatter(
			x = df_sc[0],
		    y = df_sc[1],
		    text = df.comment,
		    mode='markers',
		    marker=dict(
		        size='8',
		        color = df_sc.index,
		        colorscale='Viridis',
		        showscale=True
		    )
		)
		data = [trace1]

		py.iplot(data, filename='scatter-plot-with-colorscale')


	def tag_comment(self, start = 0, end = None):
		if end == None:
			end = int(len(self.__df)*.1)
		for i in range(start,end):
		    print(self.__df.content.loc[i])
		    print(self.__df_tag[i][self.__df_tag[i] != 0])
		    print(self.__df_attrharr[i][self.__df_attrharr[i] != 0])
		    print('---------------------------------------------------------------------')


	

	def wcloud(self, style = 'cloud', tag = None, weight = 0, tfidf = False, q = False, key = False, keyw = 1.7):
		class SimpleGroupedColorFunc(object):

		    def __init__(self, color_to_words, default_color):
		        self.word_to_color = {word: color
		                              for (color, words) in color_to_words.items()
		                              for word in words}

		        self.default_color = default_color

		    def __call__(self, word, **kwargs):
		        return self.word_to_color.get(word, self.default_color)


		class GroupedColorFunc(object):

		    def __init__(self, color_to_words, default_color):
		        self.color_func_to_words = [
		            (get_single_color_func(color), set(words))
		            for (color, words) in color_to_words.items()]

		        self.default_color_func = get_single_color_func(default_color)

		    def get_color_func(self, word):
		        
		        try:
		            color_func = next(
		                color_func for (color_func, words) in self.color_func_to_words
		                if word in words)
		        except StopIteration:
		            color_func = self.default_color_func

		        return color_func

		    def __call__(self, word, **kwargs):
		        return self.get_color_func(word)(word, **kwargs)

		if style == 'AF':
			mask = np.array(Image.open("AF.png"))
		else:
			mask = np.array(Image.open("cloud.png"))


		dct = None
		tri_corp = None
		if tag == None:
			tri_corp = self.get_trigram_corp()
			dct = Dictionary(tri_corp)
		else:
			tri_corp = self.get_trigram_corp()
			df_tag = self.get_tag()
			m = list(df_tag[list(df_tag.columns[df_tag.loc[tag] > weight])].columns)
			z = []
			for i in m:
			    z.append(tri_corp[i])
			dct = Dictionary(z)

		wgen = None
		if tfidf:
			corpus = [dct.doc2bow(line) for line in tri_corp]
			model = TfidfModel(corpus, dictionary=dct)
			new = {}
			for i in dct.token2id.keys():
			    new[i] = 0
			vector = [model[corp] for corp in corpus]
			for i in vector:
			    for j in i:
			        new[model.id2word[j[0]]] += j[1]
			wgen = new
		else:
			wgen = dct.token2id



		dict_tag_new = {}
		dict_tag = TagData.get_dict_tag()
		dict_key = list(dict_tag.keys())
		if key:
			for i in wgen.keys():
				if i in dict_key:
					wgen[i] *= keyw


		for i in wgen.keys():
			k = ''
			if '_' in i:
				for j in i:
					if j == '_':
						k += ' '
					else:
						k += j
			dict_tag_new[k] = wgen[i]
		

		stopwords = ['чуть ли', 'зачем нужный', 'не не', '24 час']
		for i in stopwords:
			try:
				dict_tag_new.pop(i)
			except:
					0

		if q:
			df_dict = pd.DataFrame(list(dict_tag_new.values()),list(dict_tag_new.keys()))
			df_dict = df_dict.sort_values(by=[0])[int(len(df_dict)*.05):int(len(df_dict) - len(df_dict)*.05)]
			dict_tag_new = {}
			for i in df_dict.index:
				dict_tag_new[i] = df_dict.loc[i].as_matrix()[0]

		wc = WordCloud(collocations=False, mask=mask, background_color='white').fit_words(dict_tag_new)


		color_to_words = {
		    '#00ff00': [],
		    'red': [],
		    'yellow': []
		    }





		for i in dict_tag.keys():
			k = ''
			if '_' in i:
				for j in i:
					if j == '_':
						k += ' '
					else:
						k += j
			else:
				k = i
			if dict_tag[i][2] == 'х/н' or dict_tag[i][2] == 'х/н/к':
				color_to_words['red'].append(k)
			elif dict_tag[i][2] == 'х/п' or dict_tag[i][2] == 'х/п/к':
				color_to_words['#00ff00'].append(k)
			elif dict_tag[i][2] == 'а':
				color_to_words['yellow'].append(k)

		default_color = 'blue'

		grouped_color_func = GroupedColorFunc(color_to_words, default_color)

		wc.recolor(color_func=grouped_color_func)

		plt.figure(figsize=[32,18])
		plt.imshow(wc, interpolation="bilinear")
		plt.axis("off")
		plt.show()