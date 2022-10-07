import os
import glob
import string
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu

def preprocessing(corpus):
	table  = str.maketrans('', '', string.punctuation)
	corpus = [document.translate(table) for document in corpus]
	corpus = [word_tokenize(document) for document in corpus]
	return corpus

def compute_metrics(references, outputs):
	precisions, recalls, f1s, blues = [], [], [], []

	for ref, so in zip(references, outputs):
		precision = len(list(set(ref) & set(so))) / len(so)
		recall    = len(list(set(ref) & set(so))) / len(ref)
		f1        = (precision * recall) / ((precision + recall) / 2) if precision + recall != 0 else -1
		blue      = sentence_bleu([ref], so, weights = [1])

		precisions.append(precision)
		recalls.append(recall)
		f1s.append(f1)
		blues.append(blue)

	precision = np.round(np.mean(precisions), 3)
	recall    = np.round(np.mean(recalls), 3)
	f1        = np.round(np.mean(f1s), 3)
	blue      = np.round(np.mean(blues), 3)
	
	return precision, recall, f1, blue

PATH_TO_DATA = 'data/'
PATH_TO_REF  = os.path.join(PATH_TO_DATA, 'references',     'newstest2020-ruen-ref.en.txt')
PATH_TO_SRC  = os.path.join(PATH_TO_DATA, 'sources',        'newstest2020-ruen-src.ru.txt')
PATH_TO_SOUT = os.path.join(PATH_TO_DATA, 'system-outputs', 'ru-en', '*.txt')

print(PATH_TO_REF)
print(PATH_TO_SRC)
print(PATH_TO_SOUT)

with open(PATH_TO_REF, 'r') as file: references = file.read().splitlines() 
references = preprocessing(references)

finals = []
for idx, so in enumerate(sorted(glob.glob(PATH_TO_SOUT))):
	with open(so, 'r') as file: system_outputs = file.read().splitlines() 
	print(f"SO: [{so}] ({idx + 1}/{len(glob.glob(PATH_TO_SOUT))})")
	system_outputs = preprocessing(system_outputs)
	precision, recall, f1, blue = compute_metrics(references, system_outputs)
	print(f"Precision: {precision}, Recall: {recall}, F1: {f1}, Blue: {blue}")
	finals.append((so, precision))

finals = sorted(finals, key = lambda tup: tup[1], reverse = True)
for so, precision in finals:
	print(f"System Outputs: {so}, with Precision: {precision}")