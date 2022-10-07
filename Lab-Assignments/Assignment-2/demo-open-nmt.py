import os

def download_and_generate_data():
	os.system('mkdir data models')
	os.system('wget https://object.pouta.csc.fi/OPUS-Europarl/v8/moses/en-ro.txt.zip -P data/')
	os.system('unzip data/en-ro.txt.zip -d data/')
	os.system('rm data/Europarl.en-ro.xml data/LICENSE data/README data/en-ro.txt.zip')
	os.system('paste data/Europarl.en-ro.ro data/Europarl.en-ro.en | shuf > data/shuf-Europarl.en-ro.both')
	os.system('rm data/Europarl.en-ro.en data/Europarl.en-ro.ro')
	os.system('sed -n 1,20000p data/shuf-Europarl.en-ro.both | cut -f 1 > data/train.ro.txt')
	os.system('sed -n 1,20000p data/shuf-Europarl.en-ro.both | cut -f 2 > data/train.en.txt')
	os.system('sed -n 20001,21000p data/shuf-Europarl.en-ro.both | cut -f 1 > data/dev.ro.txt')
	os.system('sed -n 20001,21000p data/shuf-Europarl.en-ro.both | cut -f 2 > data/dev.en.txt')
	os.system('sed -n 21001,21500p data/shuf-Europarl.en-ro.both | cut -f 1 > data/test.ro.txt')
	os.system('sed -n 21001,21500p data/shuf-Europarl.en-ro.both | cut -f 2 > data/test.en.txt')
	os.system('rm data/shuf-Europarl.en-ro.both')

def install_opennmt():
	os.system('pip install OpenNMT-py')

def run_opennmt():
	os.system('onmt_build_vocab -config config.yaml -n_sample 20000')
	os.system('onmt_train -config config.yaml')
	os.system('onmt_translate -model models/model_step_10000.pt -src data/test.en.txt -output data/pred_1000.txt -gpu 0 -verbose')

def compute_blue(PATH_TO_PREDICTIONS: str = 'data/pred_1000.txt', PATH_TO_TEST_SET: str = 'data/test.ro.txt'):
	os.system('git clone https://github.com/DataTurks-Engg/Neural_Machine_Translation.git')
	print(f"[{PATH_TO_PREDICTIONS}] -> [{PATH_TO_TEST_SET}] BLEU: ")
	os.system(f'python Neural_Machine_Translation/calculatebleu.py "{PATH_TO_PREDICTIONS}" "{PATH_TO_TEST_SET}"')
	os.system('rm bleu_out.txt')
	os.system('rm -rf Neural_Machine_Translation/')


download_and_generate_data()
install_opennmt()
run_opennmt()
compute_blue()