import torch
import torch.nn as nn
from torch.autograd import Variable 

import math, random, sys
from optparse import OptionParser
from collections import deque

import rdkit
import rdkit.Chem as Chem

#export PTHONPATH="/Users/nataliemaus/Documents/GitHub/JTNN_code"
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/Users/nataliemaus/Documents/GitHub/JTNN_code')
sys.path.insert(2, '/Users/nataliemaus/Documents/GitHub/JTNN_code/jtnn')
from jtnn import *  

#cpu 
cpu_only = True
#print("cuda?", torch.cuda.is_available()) #False 

# For molecule reconstruction, run:   
''' python3.7 reconstruct.py --test ../data/zinc/test.txt --vocab ../data/zinc/vocab.txt \
--hidden 450 --depth 3 --latent 56 \
--model MPNVAE-h450-L56-d3-beta0.005/model.iter-4 '''

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = OptionParser()
parser.add_option("-t", "--test", dest="test_path")
parser.add_option("-v", "--vocab", dest="vocab_path")
parser.add_option("-m", "--model", dest="model_path")
parser.add_option("-w", "--hidden", dest="hidden_size", default=200)
parser.add_option("-l", "--latent", dest="latent_size", default=56)
parser.add_option("-d", "--depth", dest="depth", default=3)
parser.add_option("-e", "--stereo", dest="stereo", default=1)
opts,args = parser.parse_args()

vocab = [x.strip("\r\n ") for x in open(opts.vocab_path)] 
vocab = Vocab(vocab)

hidden_size = int(opts.hidden_size)
latent_size = int(opts.latent_size)
depth = int(opts.depth)

#default: `--stereo 0` means the model will not infer stereochemistry 
#(because molecules in MOSES dataset does not contain stereochemistry).
stereo = True if int(opts.stereo) == 1 else False

#load trained VAE 
model = JTNNVAE(vocab, hidden_size, latent_size, depth, stereo=stereo)
# If you are running on a CPU-only machine, 
# please use torch.load with map_location=torch.device('cpu')
if cpu_only == True: 
    model.load_state_dict(torch.load(opts.model_path, map_location=torch.device('cpu')))
else: 
    model.load_state_dict(torch.load(opts.model_path))
    model = model.cuda()

#grab test data 
data = []
with open(opts.test_path) as f:
    for line in f:
        s = line.strip("\r\n ").split()[0]
        data.append(s)

# compute % of molcules that are decoded exactly same
# as input molecule
acc = 0.0
tot = 0
for smiles in data:
    mol = Chem.MolFromSmiles(smiles)
    smiles3D = Chem.MolToSmiles(mol, isomericSmiles=True)
    
    # VAE has method reconstruct which I assume 
    # encodes and then decodes input molecule 
    dec_smiles = model.reconstruct(smiles3D)
    #if the decoded molecule is exactly the same, increment accuracy
    if dec_smiles == smiles3D:
        acc += 1
    tot += 1
    print( acc / tot)
    """
    dec_smiles = model.recon_eval(smiles3D)
    tot += len(dec_smiles)
    for s in dec_smiles:
        if s == smiles3D:
            acc += 1
    print acc / tot
    """
