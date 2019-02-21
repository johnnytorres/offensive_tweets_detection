import os
import csv
import argparse
import logging
import unicodedata
import nltk
import pandas as pd
import unidecode

from tqdm import tqdm
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import wordpunct_tokenize, word_tokenize
from nltk.tokenize import TweetTokenizer
from preprocessor import tokenize, preprocess, clean

import preprocessor
# preprocessor.set_options(
#     preprocessor.OPT.URL,
#     preprocessor.OPT.SMILEY,
#     preprocessor.OPT.MENTION,
#     preprocessor.OPT.EMOJI,
#     preprocessor.OPT.NUMBER
# )

nltk.download('stopwords')

class DataTokenizer:
	def __init__(self, args):
		self.args = args
		self.input_file = os.path.expanduser(args.input_file)
		self.output_file = os.path.expanduser(args.output_file)
		self.text_field = args.text_field
		self.lowercase = True
		self.tokenizer = args.tokenizer  # 0: not tokenize, 1: word_tokenize, 2: punk tokenize, 3: twitter tokenizer
		self.stem = False
		self.lemmatize = False
		self.stemmer = SnowballStemmer("english") if self.stem else None
		self.lemmatizer = WordNetLemmatizer() if self.lemmatize else None
		self.tweetokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)

		stopword_list = nltk.corpus.stopwords.words(args.language)
		if 'no' in stopword_list:
			stopword_list.remove('no')
		if 'not' in stopword_list:
			stopword_list.remove('not')
		self.stopword_list = stopword_list

		if self.tokenizer==3:
			self.tweet_tokenizer = TweetTokenizer()
		
	def tokenize_short_text(self, raw_short_text):

		short_text = raw_short_text
		short_text = short_text.strip()
		short_text = unidecode.unidecode(short_text)
		
		if self.lowercase:
			short_text = short_text.lower()
		
		if self.tokenizer > 0:
			if self.tokenizer == 1:
				uttterance_tokens = word_tokenize(short_text)
			if self.tokenizer == 2:
				uttterance_tokens = wordpunct_tokenize(short_text)
			if self.tokenizer == 3:
				uttterance_tokens = self.tweet_tokenizer.tokenize(short_text)
			if self.tokenizer == 4:
				short_text = clean(short_text)
				short_text = self.remove_accented_chars(short_text)
				uttterance_tokens = self.tweetokenizer.tokenize(short_text)
				uttterance_tokens = self.remove_duplicated_sequential_words(uttterance_tokens)
				uttterance_tokens = self.remove_stopwords(uttterance_tokens)

			if self.stem:
				uttterance_tokens = [list(map(self.stemmer.stem, sub)) for sub in uttterance_tokens]
			if self.lemmatize:
				uttterance_tokens = [[self.lemmatizer.lemmatize(tok, pos='v') for tok in sub] for sub in uttterance_tokens]
			
			short_text = " ".join(uttterance_tokens)
		
		return short_text

	def remove_stopwords(self, tokens):
		filtered_tokens = [token for token in tokens if token not in self.stopword_list]
		return filtered_tokens

	def remove_duplicated_sequential_words(self, uttterance_tokens):
		i = 0
		while i < len(uttterance_tokens):
			j = i + 1
			while j < len(uttterance_tokens):
				if uttterance_tokens[i] == uttterance_tokens[j]:
					del uttterance_tokens[j]
				else:
					break
			i += 1
		return uttterance_tokens

	def remove_accented_chars(self, text):
		text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

		return text
	
	def run(self):
		ds=pd.read_csv(
			self.input_file,
			sep='\t'
		)
		desc = os.path.split(self.input_file)[1]
		comments_tokenized = []
		for i, row in tqdm(ds.iterrows(), 'tokenize {}'.format(desc), total=ds.shape[0]):
			comments_tokenized.append(self.tokenize_short_text(row[self.text_field]))
			
		ds[self.text_field] = comments_tokenized
		output_file = os.path.expanduser(self.output_file)
		ds.to_csv(output_file, index=False, sep='\t')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input-file', type=lambda x: os.path.expanduser(x))
	parser.add_argument('--output-file')
	parser.add_argument('--language')
	parser.add_argument('--text-field', type=lambda x: os.path.expanduser(x))
	parser.add_argument('--tokenizer', type=int, default=4)
	
	logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
	
	builder = DataTokenizer(parser.parse_args())
	builder.run()
