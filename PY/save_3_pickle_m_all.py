import numpy as np
import pandas as pd
import time, argparse#, json, pickle
from Bio import SeqIO, SeqUtils, motifs
from Bio.Seq import Seq 
from Bio.SeqRecord import SeqRecord
import scipy.ndimage
import scipy.stats as stats
from SMsplice import *
import pickle


# time python save_3_pickle.py -c ./canonical_datasets/canonical_dataset_GRCz11.txt -a ./allSS_datasets/allSS_dataset_GRCz11.txt -g ./GRCz11.fa -m ./maxEnt_models/zebrafish/ --prelearned_sres zebrafish --print_predictions


my_seed = 1
np.random.seed(my_seed)
print(f"seed = {my_seed}")

startTime = time.time()

def parse_arguments():
    #argument parsing 
    parser = argparse.ArgumentParser(description="parsing arguments")
    parser.add_argument("-c", "--canonical_ss", required=True)
    parser.add_argument("-a", "--all_ss", required=True)
    parser.add_argument("-g", "--genome", required=True)
    parser.add_argument("-m", "--maxent_dir", required=True)
    
    parser.add_argument("--prelearned_sres", choices = ['none', 'human', 'mouse', 'zebrafish', 'fly', 'moth', 'arabidopsis'], default = 'none')
    parser.add_argument("--learn_sres", action="store_true")
    parser.add_argument("--max_learned_scores", type=int, default = 1000)
    
    parser.add_argument("--learning_seed", choices = ['none', 'real-decoy'], default = 'none')
    parser.add_argument("--max_learned_weights", type=int, default = 15)
    
    parser.add_argument("--print_predictions", action="store_true")
    parser.add_argument("--print_local_scores", action="store_true")
    
    opts = parser.parse_args()
    return opts

args = parse_arguments()

# Load data from arguments
maxEntDir = args.maxent_dir
genome = SeqIO.to_dict(SeqIO.parse(args.genome, "fasta"))
canonical = pd.read_csv(args.canonical_ss, sep = '\t', engine='python', index_col=0, header = None)
canonical.index = canonical.index.map(str)
allSS = pd.read_csv(args.all_ss, sep = '\t', engine='python', index_col=0, header = None) 
allSS.index = allSS.index.map(str)

genes = {}
for gene in canonical.index:
    txstart = canonical.loc[gene,4]-1
    txend = canonical.loc[gene,5]
    chrom = canonical.loc[gene,2][3:]
    if not np.array_equiv(['a', 'c', 'g', 't'],
         np.unique(genome[chrom][canonical.loc[gene,4]-1:canonical.loc[gene,5]].seq.lower())): 
#         print(gene, "nonstandard base")
        continue
    
    name = gene
    geneID = gene
    description = gene + ' GeneID:'+gene + ' TranscriptID:Canonical' + ' Chromosome:' + canonical.loc[gene,2] + \
    ' Start:'+str(txstart) + ' Stop:'+str(txend) + ' Strand:'+canonical.loc[gene,3]

    if canonical.loc[gene,3] == '-':
        seq = genome[chrom][canonical.loc[gene,4]-1:canonical.loc[gene,5]].seq.reverse_complement()
    elif canonical.loc[gene,3] == '+': 
        seq = genome[chrom][canonical.loc[gene,4]-1:canonical.loc[gene,5]].seq
    else:
        print(gene, "strand error")
    
    genes[gene] = SeqRecord(seq, name = name, id = geneID, description = description)

# Additional parameters
sreEffect = 80
sreEffect3_intron = sreEffect + 19
sreEffect3_exon = sreEffect + 3
sreEffect5_intron = sreEffect + 5
sreEffect5_exon = sreEffect + 3
np.seterr(divide='ignore')

kmer = 6
E = 0
I = 1
B5 = 5
B3 = 3
train_size = 4000
test_size = 1000
score_learning_rate = .01

# Get training, validation, generalization, and test
# testGenes = canonical[(canonical[1] == 0)&canonical[2].isin(['chr2', 'chr4'])].index 
testGenes = canonical[(canonical[1] == 0)&canonical[2].isin(['chr1', 'chr3', 'chr5', 'chr7', 'chr9'])].index 
testGenes = np.intersect1d(testGenes, list(genes.keys()))

trainGenes = canonical.index
trainGenes = np.intersect1d(trainGenes, list(genes.keys()))
trainGenes = np.setdiff1d(trainGenes, testGenes)

lengthsOfGenes = np.array([len(str(genes[gene].seq)) for gene in trainGenes])
trainGenes = trainGenes[lengthsOfGenes > sreEffect3_intron]
lengthsOfGenes = np.array([len(str(genes[gene].seq)) for gene in trainGenes])
validationGenes = trainGenes[lengthsOfGenes < 200000]

generalizationGenes = np.intersect1d(validationGenes, canonical[canonical[1] == 0].index)
if len(generalizationGenes) > test_size: generalizationGenes = np.random.choice(generalizationGenes, test_size, replace = False)
validationGenes = np.setdiff1d(validationGenes, generalizationGenes)
if len(validationGenes) > test_size: validationGenes = np.random.choice(validationGenes, test_size, replace = False)

trainGenes = np.setdiff1d(trainGenes, generalizationGenes)
trainGenes = np.setdiff1d(trainGenes, validationGenes)
if len(trainGenes) > train_size: trainGenes = np.random.choice(trainGenes, train_size, replace = False)

cannonical_annotations = {}
annotations = {}
for gene in genes.keys():
    annnotation = []
    info = genes[gene].description.split(' ')
    
    if canonical.loc[gene,6] == ',': 
        cannonical_annotations[gene] = [(0, int(info[5][5:]) - int(info[4][6:]) - 1)]
        continue
        
    exonEnds = [int(start)-1 for start in canonical.loc[gene,6].split(',')[:-1]] + [int(info[5][5:])] # intron starts -> exon ends
    exonStarts = [int(info[4][6:])] + [int(end)+1 for end in canonical.loc[gene,7].split(',')[:-1]] # exon starts -> intron ends
    exonEnds[-1] -= 1
    exonStarts[0] += 2
    
    if info[6] == 'Strand:-': 
        stop = int(info[5][5:])
        for i in range(len(exonEnds),0,-1):
            annnotation.append((stop - exonEnds[i-1] - 1, stop - exonStarts[i-1] + 1))        
        
    elif info[6] == 'Strand:+': 
        start = int(info[4][6:])
        for i in range(len(exonEnds)):
            annnotation.append((exonStarts[i] - start - 2, exonEnds[i] - start))
            
    cannonical_annotations[gene] = annnotation
    annotations[gene] = {'Canonical' : annnotation}
trueSeqs = trueSequencesCannonical(genes, cannonical_annotations, E, I, B3, B5) 

# Structural parameters
numExonsPerGene, lengthSingleExons, lengthFirstExons, lengthMiddleExons, lengthLastExons, lengthIntrons \
    = structuralParameters(trainGenes, annotations)
    
N = int(np.ceil(max(numExonsPerGene)/10.0)*10)
numExonsHist = np.histogram(numExonsPerGene, bins = N, range = (0,N))[0] #nth bin is [n-1,n) bun Nth bin is [N-1,N]
numExonsDF = pd.DataFrame(list(zip(np.arange(0, N), numExonsHist)),columns = ['Index', 'Count'])

p1E = float(numExonsDF['Count'][1])/numExonsDF['Count'].sum()
pEO = float(numExonsDF['Count'][2:].sum()) / (numExonsDF['Index'][2:]*numExonsDF['Count'][2:]).sum()
numExonsDF['Prob'] = [pEO*(1-pEO)**(i-2) for i in range(0,N)] 
numExonsDF.loc[1,'Prob'] = p1E
numExonsDF.loc[0,'Prob'] = 0
numExonsDF['Empirical'] = numExonsDF['Count']/numExonsDF['Count'].sum() 
transitions = [1 - p1E, 1 - pEO]

N = 5000000
lengthIntronsHist = np.histogram(lengthIntrons, bins = N, range = (0,N))[0] # nth bin is [n-1,n) bun Nth bin is [N-1,N]
lengthIntronsDF = pd.DataFrame(list(zip(np.arange(0, N), lengthIntronsHist)), columns = ['Index', 'Count'])
lengthIntronsDF['Prob'] = lengthIntronsDF['Count']/lengthIntronsDF['Count'].sum()
pIL = geometric_smooth_tailed(lengthIntrons, N, 15, 1000, lower_cutoff=60)

lengthFirstExonsHist = np.histogram(lengthFirstExons, bins = N, range = (0,N))[0] # nth bin is [n-1,n) bun Nth bin is [N-1,N]
lengthFirstExonsDF = pd.DataFrame(list(zip(np.arange(0, N), lengthFirstExonsHist)), columns = ['Index', 'Count'])
lengthFirstExonsDF['Prob'] = lengthFirstExonsDF['Count']/lengthFirstExonsDF['Count'].sum()
pELF = adaptive_kde_tailed(lengthFirstExons, N)

lengthMiddleExonsHist = np.histogram(lengthMiddleExons, bins = N, range = (0,N))[0] # nth bin is [n-1,n) bun Nth bin is [N-1,N]
lengthMiddleExonsDF = pd.DataFrame(list(zip(np.arange(0, N), lengthMiddleExonsHist)), columns = ['Index', 'Count'])
lengthMiddleExonsDF['Prob'] = lengthMiddleExonsDF['Count']/lengthMiddleExonsDF['Count'].sum()
pELM = adaptive_kde_tailed(lengthMiddleExons, N)

lengthLastExonsHist = np.histogram(lengthLastExons, bins = N, range = (0,N))[0] # nth bin is [n-1,n) bun Nth bin is [N-1,N]
lengthLastExonsDF = pd.DataFrame(list(zip(np.arange(0, N), lengthLastExonsHist)), columns = ['Index', 'Count'])
lengthLastExonsDF['Prob'] = lengthLastExonsDF['Count']/lengthLastExonsDF['Count'].sum()
pELL = adaptive_kde_tailed(lengthLastExons, N)
pELS = 0*np.copy(pELL)

sreScores_exon = np.ones(4**kmer)
sreScores_intron = np.ones(4**kmer)

if args.prelearned_sres != 'none':
    sreScores_exon = np.load('organism_parameters/' + args.prelearned_sres + '/sreScores_exon.npy')
    sreScores_intron = np.load('organism_parameters/' + args.prelearned_sres + '/sreScores_intron.npy')

sreScores3_exon = np.copy(sreScores_exon)
sreScores5_exon = np.copy(sreScores_exon)
sreScores3_intron = np.copy(sreScores_intron)
sreScores5_intron = np.copy(sreScores_intron)


# Filter test set
lengthsOfGenes = np.array([len(str(genes[gene].seq)) for gene in testGenes])
testGenes = testGenes[lengthsOfGenes > sreEffect3_intron]

notShortIntrons = []
for gene in testGenes:
    trueThrees = np.nonzero(trueSeqs[gene] == B3)[0]
    trueFives = np.nonzero(trueSeqs[gene] == B5)[0]
    
    n_fives = np.sum(trueSeqs[gene] == B5)
    n_threes = np.sum(trueSeqs[gene] == B3)
    
    if n_fives != n_threes: notShortIntrons.append(False)
    elif np.min(trueThrees-trueFives+1) < 25: notShortIntrons.append(False)
    else: notShortIntrons.append(True)
    
notShortIntrons = np.array(notShortIntrons)
testGenes = testGenes[notShortIntrons]
lengths = np.array([len(str(genes[gene].seq)) for gene in testGenes])
sequences = [str(genes[gene].seq) for gene in testGenes]

exonicSREs5s = np.zeros((len(lengths), max(lengths)-kmer+1))
exonicSREs3s = np.zeros((len(lengths), max(lengths)-kmer+1))
intronicSREs5s = np.zeros((len(lengths), max(lengths)-kmer+1))
intronicSREs3s = np.zeros((len(lengths), max(lengths)-kmer+1))

for g, sequence in enumerate(sequences):
    exonicSREs5s[g,:lengths[g]-kmer+1] = np.log(np.array(sreScores_single(sequence.lower(), sreScores5_exon, kmer)))
    exonicSREs3s[g,:lengths[g]-kmer+1] = np.log(np.array(sreScores_single(sequence.lower(), sreScores3_exon, kmer)))
    intronicSREs5s[g,:lengths[g]-kmer+1] = np.log(np.array(sreScores_single(sequence.lower(), sreScores5_intron, kmer)))
    intronicSREs3s[g,:lengths[g]-kmer+1] = np.log(np.array(sreScores_single(sequence.lower(), sreScores3_intron, kmer)))





pred_all = viterbi(sequences = sequences, transitions = transitions, pIL = pIL, pELS = pELS, pELF = pELF, pELM = pELM, pELL = pELL, exonicSREs5s = exonicSREs5s, exonicSREs3s = exonicSREs3s, intronicSREs5s = intronicSREs5s, intronicSREs3s = intronicSREs3s, k = kmer, sreEffect5_exon = sreEffect5_exon, sreEffect5_intron = sreEffect5_intron, sreEffect3_exon = sreEffect3_exon, sreEffect3_intron = sreEffect3_intron, meDir = maxEntDir)


# import pickle

# # Combine all the variables into a dictionary
data = {
    'pred_all': pred_all,
}

# Save the dictionary to a pickle file
with open(f'm_pred_all_new_{my_seed}.pkl', 'wb') as f:
    pickle.dump(data, f)
 


f1_score = True

predFives_all = []
predThrees_all = []
trueFives_all = []
trueThrees_all = []


if (f1_score):
    # Get the Sensitivity and Precision
    num_truePositives = 0
    num_falsePositives = 0
    num_falseNegatives = 0

    for g, gene in enumerate(testGenes):
        L = lengths[g]
        predThrees = np.nonzero(pred_all[0][g,:L] == 3)[0]
        trueThrees = np.nonzero(trueSeqs[gene] == B3)[0]

        predFives = np.nonzero(pred_all[0][g,:L] == 5)[0]
        trueFives = np.nonzero(trueSeqs[gene] == B5)[0]
        
        if args.print_predictions: 
            print(gene)
            print("\tAnnotated Fives:", trueFives, "Predicted Fives:", predFives)   
            print("\tAnnotated Threes:", trueThrees, "Predicted Threes:", predThrees)
            predFives_all.append(predFives)
            predThrees_all.append(predThrees)
            trueFives_all.append(trueFives)
            trueThrees_all.append(trueThrees)     

        # num_truePositives += len(np.intersect1d(predThrees, trueThrees)) + len(np.intersect1d(predFives, trueFives))
        # num_falsePositives += len(np.setdiff1d(predThrees, trueThrees)) + len(np.setdiff1d(predFives, trueFives))
        # num_falseNegatives += len(np.setdiff1d(trueThrees, predThrees)) + len(np.setdiff1d(trueFives, predFives))

    # ssSens = num_truePositives / (num_truePositives + num_falseNegatives)
    # ssPrec = num_truePositives / (num_truePositives + num_falsePositives)
    # f1 = 2 / (1/ssSens + 1/ssPrec)
    # print("Final Test Metrics", "Recall", ssSens, "Precision", ssPrec, "f1", f1) 
    


import pickle

# Combine the prediction lists into a dictionary
data = {
    'predFives_all': predFives_all,
    'predThrees_all': predThrees_all,
    'trueFives_all': trueFives_all,
    'trueThrees_all': trueThrees_all
}


with open(f'm_predictions_new_{my_seed}.pkl', 'wb') as f:
    pickle.dump(data, f)

print("Prediction data saved successfully.")


#############
#############
#############
#############
#############
#############

transitions = np.log(transitions)
pIL = np.log(pIL)
pELS = np.log(pELS)
pELF = np.log(pELF)
pELM = np.log(pELM)
pELL = np.log(pELL)


pME = transitions[0]
p1E = np.log(1 - np.exp(pME))
pEE = transitions[1]
pEO = np.log(1 - np.exp(pEE))



emissions5, emissions3 = emissions(sequences = sequences, exonicSREs5s = exonicSREs5s, exonicSREs3s = exonicSREs3s, intronicSREs5s = intronicSREs5s, intronicSREs3s = intronicSREs3s, k = kmer, sreEffect5_exon = sreEffect5_exon, sreEffect5_intron = sreEffect5_intron, sreEffect3_exon = sreEffect3_exon, sreEffect3_intron = sreEffect3_intron, meDir = maxEntDir)


# Combine all the variables into a dictionary
import pickle

# 建立包含所有變數的字典
data = {
    'sequences': sequences,
    'pME': pME,
    'pELF': pELF,
    'pIL': pIL,
    'pEE': pEE,
    'pELM': pELM,
    'pEO': pEO,
    'pELL': pELL,
    'emissions5': emissions5,
    'emissions3': emissions3,
    'lengths': lengths,
    'trueSeqs': trueSeqs,
    'testGenes': testGenes,
    'B3': B3,
    'B5': B5,
    'transitions': transitions,
    'pELS': pELS,
    'exonicSREs5s': exonicSREs5s,
    'exonicSREs3s': exonicSREs3s,
    'intronicSREs5s': intronicSREs5s,
    'intronicSREs3s': intronicSREs3s,
    'kmer': kmer,
    'sreEffect5_exon': sreEffect5_exon,
    'sreEffect5_intron': sreEffect5_intron,
    'sreEffect3_exon': sreEffect3_exon,
    'sreEffect3_intron': sreEffect3_intron,
    'maxEntDir': maxEntDir
}

# 將字典儲存到 pickle 檔案
with open(f'm_new.pkl_{my_seed}', 'wb') as f:
    pickle.dump(data, f)









#############

# # Get the Sensitivity and Precision
# num_truePositives = 0
# num_falsePositives = 0
# num_falseNegatives = 0

# for g, gene in enumerate(testGenes):
#     L = lengths[g]
#     predThrees = np.nonzero(pred_all[0][g,:L] == 3)[0]
#     trueThrees = np.nonzero(trueSeqs[gene] == B3)[0]

#     predFives = np.nonzero(pred_all[0][g,:L] == 5)[0]
#     trueFives = np.nonzero(trueSeqs[gene] == B5)[0]
    
#     if args.print_predictions: 
#         print(gene)
#         print("\tAnnotated Fives:", trueFives, "Predicted Fives:", predFives)
#         print("\tAnnotated Threes:", trueThrees, "Predicted Threes:", predThrees)
        
#     num_truePositives += len(np.intersect1d(predThrees, trueThrees)) + len(np.intersect1d(predFives, trueFives))
#     num_falsePositives += len(np.setdiff1d(predThrees, trueThrees)) + len(np.setdiff1d(predFives, trueFives))
#     num_falseNegatives += len(np.setdiff1d(trueThrees, predThrees)) + len(np.setdiff1d(trueFives, predFives))

# if args.print_local_scores:
#     scored_sequences_5, scored_sequences_3  = score_sequences(sequences = sequences, exonicSREs5s = exonicSREs5s, exonicSREs3s = exonicSREs3s, intronicSREs5s = intronicSREs5s, intronicSREs3s = intronicSREs3s, k = kmer, sreEffect5_exon = sreEffect5_exon, sreEffect5_intron = sreEffect5_intron, sreEffect3_exon = sreEffect3_exon, sreEffect3_intron = sreEffect3_intron, meDir = maxEntDir)
  
#     for g, gene in enumerate(testGenes):
#         L = lengths[g]
#         predThrees = np.nonzero(pred_all[0][g,:L] == 3)[0]
#         predFives = np.nonzero(pred_all[0][g,:L] == 5)[0]
        
#         print(gene, "Internal Exons:")
#         for i, three in enumerate(predThrees[:-1]):
#             five = predFives[i+1]
#             print("\t", np.log2(scored_sequences_5[g,five]) + np.log2(scored_sequences_3[g,three]) + np.log2(pELM[five - three]))
        
#         print(gene, "Introns:")
#         for i, three in enumerate(predThrees):
#             five = predFives[i]
#             print("\t", np.log2(scored_sequences_5[g,five]) + np.log2(scored_sequences_3[g,three]) + np.log2(pELM[-five + three]))


# ssSens = num_truePositives / (num_truePositives + num_falseNegatives)
# ssPrec = num_truePositives / (num_truePositives + num_falsePositives)
# f1 = 2 / (1/ssSens + 1/ssPrec)
# print("Final Test Metrics", "Recall", ssSens, "Precision", ssPrec, "f1", f1) 
 


