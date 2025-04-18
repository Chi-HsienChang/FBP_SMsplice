#distutils: extra_link_args=-fopenmp
from cython import parallel
from cython.parallel import prange
import numpy as np
import time
from sklearn.neighbors import KernelDensity
from awkde import GaussianKDE
from math import exp 
from libc.math cimport exp as c_exp
cimport openmp

# 設定 NumPy 隨機種子
np.random.seed(0)

# SMsplice_low_memory.pyx
# cython: language_level=3

import math
cimport numpy as np

###################################################
# (A) 新增：保留前k大的state
###################################################
def keep_top_k_states(state_dict, k):
    """
    只保留 state_dict 中對數機率最高的前 k 個 state。
    state_dict: { state: alpha_score }
    k: 要保留的 state 數目
    """
    if len(state_dict) <= k:
        return state_dict
    sorted_items = sorted(state_dict.items(), key=lambda x: x[1], reverse=True)
    new_dict = {}
    for i in range(k):
        st, score = sorted_items[i]
        new_dict[st] = score
    return new_dict


###################################################
# 1. 數值穩定的 logsumexp
###################################################
cdef double logsumexp(list vals):
    cdef int n = len(vals)
    if n == 0:
        return float('-inf')
    cdef double max_val = vals[0]
    cdef int i
    for i in range(1, n):
        if vals[i] > max_val:
            max_val = vals[i]
    if max_val == float('-inf'):
        return float('-inf')
    cdef double total = 0.0, x
    for i in range(n):
        x = vals[i] - max_val
        total += math.exp(x)
    return max_val + math.log(total)

###################################################
# 2. transition_dp: 狀態轉移 (與原版一致)
###################################################
cdef tuple transition_dp(
    tuple state, double log_score, int pos, int symbol, 
    object sequences, int length,
    double pME, double[:] pELF, double[:] pIL, 
    double pEE, double[:] pELM,
    double[:] emissions5, double[:] emissions3
):
    cdef:
        int used5       = state[0]
        int used3       = state[1]
        int lastSymbol  = state[2]
        int zeroCount   = state[3]
        int last5Pos    = state[4]
        int last3Pos    = state[5]

        double new_log_score = log_score
        int newUsed5         = used5
        int newUsed3         = used3
        int newZeroCount     = zeroCount
        int newLast5Pos      = last5Pos
        int newLast3Pos      = last3Pos
        int gap_5, gap_3
        int newLastSymbol

    # symbol == 0
    if symbol == 0:
        if lastSymbol == 5 or lastSymbol == 3:
            newZeroCount = zeroCount + 1
        newLastSymbol = 0

    # symbol == 5
    elif symbol == 5:
        if emissions5[pos] <= float('-inf'):
            return None
        if pos + 1 >= length:
            return None
        if not (sequences[pos] == 'G' and sequences[pos+1] == 'T'):
            return None
        if lastSymbol == 5 or (((lastSymbol == 5) or (lastSymbol == 3)) and zeroCount < 5) or (used5 != used3):
            return None

        if used5 == 0:
            new_log_score += pME + pELF[pos - 1] + emissions5[pos]
        else:
            gap_5 = (pos - last3Pos) - 2
            if gap_5 < 0 or gap_5 >= pELM.shape[0]:
                return None
            new_log_score += pEE + pELM[gap_5] + emissions5[pos]

        newUsed5 = used5 + 1
        newLast5Pos = pos
        newZeroCount = 0
        newLastSymbol = 5

    # symbol == 3
    elif symbol == 3:
        if emissions3[pos] <= float('-inf'):
            return None
        if pos - 1 < 0:
            return None
        if not (sequences[pos] == 'G' and sequences[pos-1] == 'A'):
            return None
        # 這裡的 float('-inf') 在原本可能是 -19 等定值, 改為 if zeroCount < 5, etc. 按需求調整
        if lastSymbol == 3 or (((lastSymbol == 5) or (lastSymbol == 3)) and zeroCount < 5) or (used5 != used3 + 1):
            return None

        gap_3 = (pos - last5Pos) - 2
        if gap_3 < 0 or gap_3 >= pIL.shape[0]:
            return None
        new_log_score += pIL[gap_3] + emissions3[pos]

        newUsed3 = used3 + 1
        newLast3Pos = pos
        newZeroCount = 0
        newLastSymbol = 3

    else:
        return None

    cdef tuple new_state = (newUsed5, newUsed3, newLastSymbol, newZeroCount, newLast5Pos, newLast3Pos)
    return (new_state, new_log_score)

###################################################
# 3. 單步 forward DP：僅從前一層計算下一層，並保留前 top_k
###################################################
cdef dict forward_dp_step(
    dict F_prev, 
    int pos, 
    object sequences, 
    int length,
    double pME, 
    double[:] pELF, 
    double[:] pIL, 
    double pEE, 
    double[:] pELM, 
    double[:] emissions5, 
    double[:] emissions3,
    int top_k
):
    cdef dict F_curr = {}
    cdef list allowed_symbols
    if pos == 0 or pos == length - 1:
        allowed_symbols = [0]
    else:
        allowed_symbols = [0, 5, 3]

    cdef tuple state, new_state_tuple, new_state
    cdef double alpha_score, new_log_score
    cdef int symbol

    for state, alpha_score in F_prev.items():
        if alpha_score == float('-inf') or math.isnan(alpha_score):
            continue
        for symbol in allowed_symbols:
            new_state_tuple = transition_dp(
                state, alpha_score, pos, symbol,
                sequences, length,
                pME, pELF, pIL, pEE, pELM,
                emissions5, emissions3
            )
            if new_state_tuple is None:
                continue
            new_state, new_log_score = new_state_tuple

            if new_state in F_curr:
                F_curr[new_state] = logsumexp([F_curr[new_state], new_log_score])
            else:
                F_curr[new_state] = new_log_score

    # 只保留對數機率最高的前 top_k 個 state
    F_curr = keep_top_k_states(F_curr, top_k)
    return F_curr

###################################################
# 4. forward_backward_low_memory: 利用 checkpointing + 保留top_k
###################################################
cpdef tuple forward_backward_low_memory(
    object sequences,
    double pME,
    double[:] pELF,
    double[:] pIL,
    double pEE,
    double[:] pELM,
    double pEO,
    double[:] pELL,
    double[:] emissions5,
    double[:] emissions3,
    int length,
    int checkpoint_interval,  # 例如：1000
    int top_k                 # 新增參數：每步保留前k大的state
):
    """
    此函數採用 checkpointing 技術 + 保留前k大state：
      1. 前向計算時只儲存每隔 checkpoint_interval 的 F 狀態 (且每步保留前k大),
         其餘部分後續於 backward 區段中重算。
      2. 後向計算時，將序列依 checkpoint 分成數個區段，
         針對每個區段重算 forward DP（僅該區段內, 同樣保留前k大）並做 backward 遞推，
         同時計算各 pos 的 posterior。
    回傳: (post_list, logZ)
      post_list[i] = { symbol: posterior_prob }，i 為全局序列位置。
    """
    cdef dict checkpoints = {}  
    cdef dict F_current = {}
    cdef tuple init_state = (0, 0, 0, 1, -1, -1)
    F_current[init_state] = 0.0
    checkpoints[0] = F_current.copy()

    cdef int pos
    # 前向遞推：僅在 checkpoint 位置儲存 F
    for pos in range(0, length):
        F_current = forward_dp_step(
            F_current, pos, sequences, length,
            pME, pELF, pIL, pEE, pELM, emissions5, emissions3,
            top_k
        )
        if ((pos + 1) % checkpoint_interval == 0) or (pos + 1 == length):
            checkpoints[pos + 1] = F_current.copy()

    # 此時 F_current 為 F[length]
    # 計算 B[length]
    cdef dict B_current = {}
    cdef double tail
    cdef int used5, used3, lastSymbol, last3Pos, ell_index
    for state, alpha_score in F_current.items():
        used5 = state[0]
        used3 = state[1]
        lastSymbol = state[2]
        last3Pos = state[5]
        # 可能依需求檢查條件: (used5 == used3) and ...
        if lastSymbol == 0 and (used5 == used3) and ((used5 + used3) > 0):
            tail = 0.0
            ell_index = (length - last3Pos) - 2
            if ell_index >= 0 and ell_index < pELL.shape[0]:
                tail += pEO + pELL[ell_index]
            B_current[state] = tail

    # 計算最終的 partition function logZ = logsumexp( F[length][s] + B[length][s] )
    cdef double logZ
    cdef list terminal_logs = []
    for state, alpha_score in F_current.items():
        if state in B_current:
            if (not math.isnan(alpha_score)) and (not math.isnan(B_current[state])):
                terminal_logs.append(alpha_score + B_current[state])
    if terminal_logs:
        logZ = logsumexp(terminal_logs)
    else:
        logZ = float('-inf')

    # 初始化 posterior 結果，每個位置一個 dict
    cdef list post_list = [ {} for _ in range(length) ]

    # 將 checkpoint 位置取出，排序（由小到大）
    cdef list ckpt_positions = sorted(checkpoints.keys())
    # B_next_segment 為目前區段尾端的 backward 值，初始即 B_current（對應 pos = length）
    cdef dict B_next_segment = B_current

    cdef int i, seg_start, seg_end, seg_len, j, global_pos
    cdef list seg_F 
    cdef list seg_B 
    cdef dict B_seg
    cdef list contributions
    cdef tuple new_state_tuple, new_state
    cdef double alpha_val, new_log_score
    cdef double b_val 
    cdef double val 
    cdef double prob
    cdef int sym 
    cdef list allowed_symbols

    # 從最後一個 checkpoint 區段往前處理
    for i in range(len(ckpt_positions) - 1, 0, -1):
        seg_end = ckpt_positions[i]
        seg_start = ckpt_positions[i - 1]
        seg_len = seg_end - seg_start

        # 針對此區段重新計算 forward DP（僅該區段內），一樣在每步保留前k大
        seg_F = [None] * (seg_len + 1)
        seg_F[0] = checkpoints[seg_start].copy()

        for j in range(0, seg_len):
            global_pos = seg_start + j
            seg_F[j + 1] = forward_dp_step(
                seg_F[j], global_pos, sequences, length,
                pME, pELF, pIL, pEE, pELM, 
                emissions5, emissions3,
                top_k
            )

        # 現在對該區段進行 backward 遞推：
        seg_B = [None] * (seg_len + 1)
        seg_B[seg_len] = B_next_segment
        # 從區段內由後往前
        for j in range(seg_len - 1, -1, -1):
            global_pos = seg_start + j
            B_seg = {}

            if global_pos == 0 or global_pos == length - 1:
                allowed_symbols = [0]
            else:
                allowed_symbols = [0, 5, 3]

            for state, alpha_val in seg_F[j].items():
                contributions = []
                for sym in allowed_symbols:
                    new_state_tuple = transition_dp(
                        state, alpha_val, global_pos, sym,
                        sequences, length,
                        pME, pELF, pIL, pEE, pELM,
                        emissions5, emissions3
                    )
                    if new_state_tuple is None:
                        continue
                    new_state, new_log_score = new_state_tuple
                    if new_state in seg_B[j + 1]:
                        # = new_log_score (含F的轉移) - alpha_val (去除前面F_val)
                        #   + seg_B[j+1][new_state] (後半段backward值)
                        contributions.append(new_log_score - alpha_val + seg_B[j + 1][new_state])

                if contributions:
                    B_seg[state] = logsumexp(contributions)

            seg_B[j] = B_seg

            # 利用 seg_F[j] 與 seg_B[j] 更新全局 posterior
            for state, alpha_val in seg_F[j].items():
                if state in seg_B[j]:
                    b_val = seg_B[j][state]
                    if (not math.isnan(alpha_val)) and (not math.isnan(b_val)):
                        val = alpha_val + b_val - logZ
                        if val != float('-inf') and (not math.isnan(val)):
                            prob = math.exp(val)
                            sym = state[2]
                            if sym in post_list[global_pos]:
                                post_list[global_pos][sym] += prob
                            else:
                                post_list[global_pos][sym] = prob

        # 區段處理完畢，將該區段最前端的 backward 值作為下一區段的 B_next_segment
        B_next_segment = seg_B[0]

    return post_list, logZ


############################################
# 結束
############################################
########################################
########################################
########################################
########################################
########################################
########################################
########################################
########################################
########################################
########################################
########################################
########################################
########################################
########################################
########################################
########################################
########################################
########################################
########################################
########################################
########################################
########################################
########################################
########################################
########################################
########################################
########################################
########################################
########################################
########################################
########################################
########################################
########################################
########################################
########################################
########################################
########################################
########################################
########################################



def baseToInt(str base):
    if base == 'a': return 0
    elif base == 'c': return 1
    elif base == 'g': return 2
    elif base == 't': return 3
    else:
        print("nonstandard base encountered:", base)
        return -1

def intToBase(int i):
    if i == 0: return 'a'
    elif i == 1: return 'c'
    elif i == 2: return 'g'
    elif i == 3: return 't'
    else: 
        print("nonbase integer encountered:", i)
        return ''

def hashSequence(str seq):
    cdef int i
    cdef int sum = 0 
    cdef int l = len(seq)
    for i in range(l):
        sum += (4**(l-i-1))*baseToInt(seq[i])
    return sum
    
def unhashSequence(int num, int l):
    seq = ''
    for i in range(l):
        seq += intToBase(num // 4**(l-i-1))
        num -= (num // 4**(l-i-1))*(4**(l-i-1))
    return seq
    
def trueSequencesCannonical(genes, annotations, E = 0, I = 1, B3 = 3, B5 = 5):
    # Converts gene annotations to sequences of integers indicating whether the sequence is exonic, intronic, or splice site,
    # Inputs
    #   - genes: a biopython style dictionary of the gene sequences
    #   - annotations: the splicing annotations dictionary
    #   - E, I, B3, B5: the integer indicators for exon, intron, 3'ss, and 5'ss, respectively
    trueSeqs = {}
    for gene in annotations.keys():
        if gene not in genes.keys(): 
            print(gene, 'has annotation, but was not found in the fasta file of genes') 
            continue
        
        transcript = annotations[gene]
        if len(transcript) == 1: 
            trueSeqs[gene] = np.zeros(len(genes[gene]), dtype = int) + E
            continue # skip the rest for a single exon case
        
        # First exon 
        true = np.zeros(len(genes[gene]), dtype = int) + I
        three = transcript[0][0] - 1 # Marking the beginning of the first exon
        five = transcript[0][1] + 1
        true[range(three+1, five)] = E
        true[five] = B5
        
        # Internal exons 
        for exon in transcript[1:-1]:
            three = exon[0] - 1
            five = exon[1] + 1
            true[three] = B3
            true[five] = B5
            true[range(three+1, five)] = E
            
        # Last exon 
        three = transcript[-1][0] - 1
        true[three] = B3
        five = transcript[-1][1] + 1 # Marking the end of the last exon
        true[range(three+1, five)] = E
                
        trueSeqs[gene] = true
        
    return(trueSeqs)

def trainAllTriplets(sequences, cutoff = 10**(-5)):
    # Train maximum entropy models from input sequences with triplet conditions
    train = np.zeros((len(sequences),len(sequences[0])), dtype = int)
    for (i, seq) in enumerate(sequences):
        for j in range(len(seq)):
            train[i,j] = baseToInt(seq[j])
    prob = np.log(np.zeros(4**len(sequences[0])) + 4**(-len(sequences[0])))
    Hprev = -np.sum(prob*np.exp(prob))/np.log(2)
    H = -1
    sequences = np.zeros((4**len(sequences[0]),len(sequences[0])), dtype = int)
    l = len(sequences[0]) - 1 
    for i in range(sequences.shape[1]):
        sequences[:,i] = ([0]*4**(l-i) + [1]*4**(l-i) + [2]*4**(l-i) +[3]*4**(l-i))*4**i
    while np.abs(Hprev - H) > cutoff:
        #print(np.abs(Hprev - H))
        Hprev = H
        for pos in range(sequences.shape[1]):
            for base in range(4):
                Q = np.sum(train[:,pos] == base)/float(train.shape[0])
                if Q == 0: continue
                Qhat = np.sum(np.exp(prob[sequences[:,pos] == base]))
                prob[sequences[:,pos] == base] += np.log(Q) - np.log(Qhat)
                prob[sequences[:,pos] != base] += np.log(1-Q) - np.log(1-Qhat)
                
                for pos2 in np.setdiff1d(range(sequences.shape[1]), range(pos+1)):
                    for base2 in range(4):
                        Q = np.sum((train[:,pos] == base)*(train[:,pos2] == base2))/float(train.shape[0])
                        if Q == 0: continue
                        which = (sequences[:,pos] == base)*(sequences[:,pos2] == base2)
                        Qhat = np.sum(np.exp(prob[which]))
                        prob[which] += np.log(Q) - np.log(Qhat)
                        prob[np.invert(which)] += np.log(1-Q) - np.log(1-Qhat)
                        
                        for pos3 in np.setdiff1d(range(sequences.shape[1]), range(pos2+1)):
                            for base3 in range(4):
                                Q = np.sum((train[:,pos] == base)*(train[:,pos2] == base2)*(train[:,pos3] == base3))/float(train.shape[0])
                                if Q == 0: continue
                                which = (sequences[:,pos] == base)*(sequences[:,pos2] == base2)*(sequences[:,pos3] == base3)
                                Qhat = np.sum(np.exp(prob[which]))
                                prob[which] += np.log(Q) - np.log(Qhat)
                                prob[np.invert(which)] += np.log(1-Q) - np.log(1-Qhat)
        H = -np.sum(prob*np.exp(prob))/np.log(2)
    return np.exp(prob)

def structuralParameters(genes, annotations, minIL = 0):
    # Get the empirical length distributions for introns and single, first, middle, and last exons, as well as number exons per gene
    
    # Transitions
    numExonsPerGene = [] 
    
    # Length Distributions
    lengthSingleExons = []
    lengthFirstExons = []
    lengthMiddleExons = []
    lengthLastExons = []
    lengthIntrons = []
    
    for gene in genes:
        if len(annotations[gene]) == 0: 
            print('missing annotation for', gene)
            continue
        numExons = 0
        introns = []
        singleExons = []
        firstExons = []
        middleExons = []
        lastExons = []
        
        for transcript in annotations[gene].values():
            numExons += len(transcript)
            
            # First exon 
            three = transcript[0][0] # Make three the first base
            five = transcript[0][1] + 1
            if len(transcript) == 1: 
                singleExons.append((three, five-1))
                continue # skip the rest for a single exon case
            firstExons.append((three, five-1)) # since three is the first base
            
            # Internal exons 
            for exon in transcript[1:-1]:
                three = exon[0] - 1 
                introns.append((five+1,three-1))
                five = exon[1] + 1
                middleExons.append((three+1, five-1))
                
            # Last exon 
            three = transcript[-1][0] - 1
            introns.append((five+1,three-1))
            five = transcript[-1][1] + 1
            lastExons.append((three+1, five-1))
        
        geneIntronLengths = [minIL]
        for intron in set(introns):
            geneIntronLengths.append(intron[1] - intron[0] + 1)
        
        if np.min(geneIntronLengths) < minIL: continue
        
        for intron in set(introns): lengthIntrons.append(intron[1] - intron[0] + 1)
        for exon in set(singleExons): lengthSingleExons.append(exon[1] - exon[0] + 1)
        for exon in set(firstExons): lengthFirstExons.append(exon[1] - exon[0] + 1)
        for exon in set(middleExons): lengthMiddleExons.append(exon[1] - exon[0] + 1)
        for exon in set(lastExons): lengthLastExons.append(exon[1] - exon[0] + 1)
            
        numExonsPerGene.append(float(numExons)/len(annotations[gene]))
        
    return(numExonsPerGene, lengthSingleExons, lengthFirstExons, lengthMiddleExons, lengthLastExons, lengthIntrons)

def adaptive_kde_tailed(lengths, N, geometric_cutoff = .8, lower_cutoff=0):
    adaptive_kde = GaussianKDE(alpha = 1) 
    adaptive_kde.fit(np.array(lengths)[:,None]) 
    
    lengths = np.array(lengths)
    join = np.sort(lengths)[int(len(lengths)*geometric_cutoff)] 
    
    smoothed = np.zeros(N)
    smoothed[:join+1] = adaptive_kde.predict(np.arange(join+1)[:,None])
    
    s = 1-np.sum(smoothed)
    p = smoothed[join]
    smoothed[join+1:] = np.exp(np.log(p) + np.log(s/(s+p))*np.arange(1,len(smoothed)-join))
    smoothed[:lower_cutoff] = 0
    smoothed /= np.sum(smoothed)
    
    return(smoothed)
    
def geometric_smooth_tailed(lengths, N, bandwidth, join, lower_cutoff=0):
    lengths = np.array(lengths)
    smoothing = KernelDensity(bandwidth = bandwidth).fit(lengths[:, np.newaxis]) 
    
    smoothed = np.zeros(N)
    smoothed[:join+1] = np.exp(smoothing.score_samples(np.arange(join+1)[:,None]))
    
    s = 1-np.sum(smoothed)
    p = smoothed[join]
    smoothed[join+1:] = np.exp(np.log(p) + np.log(s/(s+p))*np.arange(1,len(smoothed)-join))
    smoothed[:lower_cutoff] = 0
    smoothed /= np.sum(smoothed)
    
    return(smoothed)

def maxEnt5(geneNames, genes, dir):
    # Get all the 5'SS maxent scores for each of the genes in geneNames
    scores = {}
    prob = np.load(dir + '/maxEnt5_prob.npy')
    prob0 = np.load(dir + '/maxEnt5_prob0.npy') 
        
    for gene in geneNames:
        sequence = str(genes[gene].seq).lower()
        sequence5 = np.array([hashSequence(sequence[i:i+9]) for i in range(len(sequence)-9+1)])
        scores[gene] = np.zeros(len(sequence)) - np.inf
        scores[gene][3:-5] = np.log2(prob[sequence5]) - np.log2(prob0[sequence5])
        scores[gene] = np.exp2(scores[gene])
    
    return scores
    
def maxEnt5_single(str seq, str dir):
    prob = np.load(dir + 'maxEnt5_prob.npy')
    prob0 = np.load(dir + 'maxEnt5_prob0.npy')
    
    seq = seq.lower()
    sequence5 = np.array([hashSequence(seq[i:i+9]) for i in range(len(seq)-9+1)])
    scores = np.log2(np.zeros(len(seq)))
    scores[3:-5] = np.log2(prob[sequence5]) - np.log2(prob0[sequence5])
    return np.exp2(scores)
    
def maxEnt3(geneNames, genes, dir):
    # Get all the 3'SS maxent scores for each of the genes in geneNames
    scores = {}
    prob0 = np.load(dir + 'maxEnt3_prob0.npy')
    prob1 = np.load(dir + 'maxEnt3_prob1.npy')
    prob2 = np.load(dir + 'maxEnt3_prob2.npy')
    prob3 = np.load(dir + 'maxEnt3_prob3.npy')
    prob4 = np.load(dir + 'maxEnt3_prob4.npy')
    prob5 = np.load(dir + 'maxEnt3_prob5.npy')
    prob6 = np.load(dir + 'maxEnt3_prob6.npy')
    prob7 = np.load(dir + 'maxEnt3_prob7.npy')
    prob8 = np.load(dir + 'maxEnt3_prob8.npy')
    
    prob0_0 = np.load(dir + 'maxEnt3_prob0_0.npy')
    prob1_0 = np.load(dir + 'maxEnt3_prob1_0.npy')
    prob2_0 = np.load(dir + 'maxEnt3_prob2_0.npy')
    prob3_0 = np.load(dir + 'maxEnt3_prob3_0.npy')
    prob4_0 = np.load(dir + 'maxEnt3_prob4_0.npy')
    prob5_0 = np.load(dir + 'maxEnt3_prob5_0.npy')
    prob6_0 = np.load(dir + 'maxEnt3_prob6_0.npy')
    prob7_0 = np.load(dir + 'maxEnt3_prob7_0.npy')
    prob8_0 = np.load(dir + 'maxEnt3_prob8_0.npy')
    
    for gene in geneNames:
        sequence = str(genes[gene].seq).lower()
        sequences23 = [sequence[i:i+23] for i in range(len(sequence)-23+1)]
        hash0 = np.array([hashSequence(seq[0:7]) for seq in sequences23])
        hash1 = np.array([hashSequence(seq[7:14]) for seq in sequences23])
        hash2 = np.array([hashSequence(seq[14:]) for seq in sequences23])
        hash3 = np.array([hashSequence(seq[4:11]) for seq in sequences23])
        hash4 = np.array([hashSequence(seq[11:18]) for seq in sequences23])
        hash5 = np.array([hashSequence(seq[4:7]) for seq in sequences23])
        hash6 = np.array([hashSequence(seq[7:11]) for seq in sequences23])
        hash7 = np.array([hashSequence(seq[11:14]) for seq in sequences23])
        hash8 = np.array([hashSequence(seq[14:18]) for seq in sequences23])
        
        probs = np.log2(prob0[hash0]) + np.log2(prob1[hash1]) + np.log2(prob2[hash2]) + \
            np.log2(prob3[hash3]) + np.log2(prob4[hash4]) - np.log2(prob5[hash5]) - \
            np.log2(prob6[hash6]) - np.log2(prob7[hash7]) - np.log2(prob8[hash8]) - \
            (np.log2(prob0_0[hash0]) + np.log2(prob1_0[hash1]) + np.log2(prob2_0[hash2]) + \
            np.log2(prob3_0[hash3]) + np.log2(prob4_0[hash4]) - np.log2(prob5_0[hash5]) - \
            np.log2(prob6_0[hash6]) - np.log2(prob7_0[hash7]) - np.log2(prob8_0[hash8]))
            
        scores[gene] = np.zeros(len(sequence)) - np.inf
        scores[gene][19:-3] = probs
        scores[gene] = np.exp2(scores[gene])
    
    return scores
    
def maxEnt3_single(str seq, str dir):
    prob0 = np.load(dir + 'maxEnt3_prob0.npy')
    prob1 = np.load(dir + 'maxEnt3_prob1.npy')
    prob2 = np.load(dir + 'maxEnt3_prob2.npy')
    prob3 = np.load(dir + 'maxEnt3_prob3.npy')
    prob4 = np.load(dir + 'maxEnt3_prob4.npy')
    prob5 = np.load(dir + 'maxEnt3_prob5.npy')
    prob6 = np.load(dir + 'maxEnt3_prob6.npy')
    prob7 = np.load(dir + 'maxEnt3_prob7.npy')
    prob8 = np.load(dir + 'maxEnt3_prob8.npy')
    
    prob0_0 = np.load(dir + 'maxEnt3_prob0_0.npy')
    prob1_0 = np.load(dir + 'maxEnt3_prob1_0.npy')
    prob2_0 = np.load(dir + 'maxEnt3_prob2_0.npy')
    prob3_0 = np.load(dir + 'maxEnt3_prob3_0.npy')
    prob4_0 = np.load(dir + 'maxEnt3_prob4_0.npy')
    prob5_0 = np.load(dir + 'maxEnt3_prob5_0.npy')
    prob6_0 = np.load(dir + 'maxEnt3_prob6_0.npy')
    prob7_0 = np.load(dir + 'maxEnt3_prob7_0.npy')
    prob8_0 = np.load(dir + 'maxEnt3_prob8_0.npy')
    
    seq = seq.lower()
    sequences23 = [seq[i:i+23] for i in range(len(seq)-23+1)]
    hash0 = np.array([hashSequence(seq[0:7]) for seq in sequences23])
    hash1 = np.array([hashSequence(seq[7:14]) for seq in sequences23])
    hash2 = np.array([hashSequence(seq[14:]) for seq in sequences23])
    hash3 = np.array([hashSequence(seq[4:11]) for seq in sequences23])
    hash4 = np.array([hashSequence(seq[11:18]) for seq in sequences23])
    hash5 = np.array([hashSequence(seq[4:7]) for seq in sequences23])
    hash6 = np.array([hashSequence(seq[7:11]) for seq in sequences23])
    hash7 = np.array([hashSequence(seq[11:14]) for seq in sequences23])
    hash8 = np.array([hashSequence(seq[14:18]) for seq in sequences23])
    
    probs = np.log2(prob0[hash0]) + np.log2(prob1[hash1]) + np.log2(prob2[hash2]) + \
            np.log2(prob3[hash3]) + np.log2(prob4[hash4]) - np.log2(prob5[hash5]) - \
            np.log2(prob6[hash6]) - np.log2(prob7[hash7]) - np.log2(prob8[hash8]) - \
            (np.log2(prob0_0[hash0]) + np.log2(prob1_0[hash1]) + np.log2(prob2_0[hash2]) + \
            np.log2(prob3_0[hash3]) + np.log2(prob4_0[hash4]) - np.log2(prob5_0[hash5]) - \
            np.log2(prob6_0[hash6]) - np.log2(prob7_0[hash7]) - np.log2(prob8_0[hash8]))
            
    scores = np.log2(np.zeros(len(seq)))
    scores[19:-3] = probs
    return np.exp2(scores)

def sreScores_single(str seq, double [:] sreScores, int kmer = 6):
    indices = [hashSequence(seq[i:i+kmer]) for i in range(len(seq)-kmer+1)]
    sequenceSRES = [sreScores[indices[i]] for i in range(len(indices))]
    return sequenceSRES

def get_all_5ss(gene, reference, genes):
    # Get all the 5'SS for a gene based on the annotation in reference
    info = genes[gene].description.split(' ')
    if reference.loc[gene,6] == ',': exonEnds = []
    else: exonEnds = [int(start)-1 for start in reference.loc[gene,6].split(',')[:-1]] # intron starts -> exon ends
    if reference.loc[gene,7] == ',': exonStarts = []
    else: exonStarts = [int(end)+1 for end in reference.loc[gene,7].split(',')[:-1]] # exon starts -> intron ends
    
    if info[6] == 'Strand:-': 
        stop = int(info[5][5:])
        annnotation = [stop - exonStarts[i-1] + 2 for i in range(len(exonStarts),0,-1)]      
        
    elif info[6] == 'Strand:+': 
        start = int(info[4][6:])
        annnotation = [exonEnds[i] - start + 1 for i in range(len(exonEnds))]
        
    return(annnotation)

def get_all_3ss(gene, reference, genes):
    # Get all the 3'SS for a gene based on the annotation in reference
    info = genes[gene].description.split(' ')
    if reference.loc[gene,6] == ',': exonEnds = []
    else: exonEnds = [int(start)-1 for start in reference.loc[gene,6].split(',')[:-1]] # intron starts -> exon ends
    if reference.loc[gene,7] == ',': exonStarts = []
    else: exonStarts = [int(end)+1 for end in reference.loc[gene,7].split(',')[:-1]] # exon starts -> intron ends
    
    if info[6] == 'Strand:-': 
        stop = int(info[5][5:])
        annnotation = [stop - exonEnds[i-1] - 2 for i in range(len(exonEnds),0,-1)]      
        
    elif info[6] == 'Strand:+': 
        start = int(info[4][6:])
        annnotation = [exonStarts[i] - start - 3 for i in range(len(exonStarts))]
        
    return(annnotation)

def get_hexamer_real_decoy_counts(geneNames, trueSeqs, decoySS, genes, kmer, sreEffect5_exon, sreEffect5_intron, sreEffect3_exon, sreEffect3_intron, B3 = 3, B5 = 5):
    # Get the counts of hexamers in the flanking regions for real and decoy ss with restriction to exons and introns for the real ss
    true_counts_5_intron = np.zeros(4**kmer, dtype = np.dtype("i"))
    true_counts_5_exon = np.zeros(4**kmer, dtype = np.dtype("i"))
    true_counts_3_intron = np.zeros(4**kmer, dtype = np.dtype("i"))
    true_counts_3_exon = np.zeros(4**kmer, dtype = np.dtype("i"))
    decoy_counts_5_intron = np.zeros(4**kmer, dtype = np.dtype("i"))
    decoy_counts_5_exon = np.zeros(4**kmer, dtype = np.dtype("i"))
    decoy_counts_3_intron = np.zeros(4**kmer, dtype = np.dtype("i"))
    decoy_counts_3_exon = np.zeros(4**kmer, dtype = np.dtype("i"))
    
    for gene in geneNames:
        trueThrees = np.nonzero(trueSeqs[gene] == B3)[0][:-1]
        trueFives = np.nonzero(trueSeqs[gene] == B5)[0][1:]
        for i in range(len(trueThrees)):
            three = trueThrees[i]
            five = trueFives[i]
            
            # 3'SS
            sequence = str(genes[gene].seq[three+4:three+sreEffect3_exon+1].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[three+4:].lower())
            if five-3 < three+sreEffect3_exon+1: sequence = str(genes[gene].seq[three+4:five-3].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: true_counts_3_exon[s] += 1
            
            sequence = str(genes[gene].seq[three-sreEffect3_intron:three-19].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[:three-19].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: true_counts_3_intron[s] += 1
                
            # 5'SS
            sequence = str(genes[gene].seq[five-sreEffect5_exon:five-3].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[:five-3].lower())
            if five-sreEffect5_exon < three+4: sequence = str(genes[gene].seq[three+4:five-3].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: true_counts_5_exon[s] += 1
            
            sequence = str(genes[gene].seq[five+6:five+sreEffect5_intron+1].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[five+6:].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: true_counts_5_intron[s] += 1
        
        decoyThrees = np.nonzero(decoySS[gene] == B3)[0]
        decoyFives = np.nonzero(decoySS[gene] == B5)[0]
        for ss in decoyFives:
            sequence = str(genes[gene].seq[ss-sreEffect5_exon:ss-3].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[:ss-3].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: decoy_counts_5_exon[s] += 1
            
            sequence = str(genes[gene].seq[ss+6:ss+sreEffect5_intron+1].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[ss+6:].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: decoy_counts_5_intron[s] += 1
    
        for ss in decoyThrees:
            sequence = str(genes[gene].seq[ss+4:ss+sreEffect3_exon+1].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[ss+4:].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: decoy_counts_3_exon[s] += 1
            
            sequence = str(genes[gene].seq[ss-sreEffect3_intron:ss-19].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[:ss-19].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: decoy_counts_3_intron[s] += 1
    
    return(true_counts_5_intron, true_counts_5_exon, true_counts_3_intron, true_counts_3_exon, 
           decoy_counts_5_intron, decoy_counts_5_exon, decoy_counts_3_intron, decoy_counts_3_exon)

def get_hexamer_counts(geneNames, set1, set2, genes, kmer, sreEffect5_exon, sreEffect5_intron, sreEffect3_exon, sreEffect3_intron, B3 = 3, B5 = 5):
    # Get the counts of hexamers in the flanking regions for two sets of ss
    set1_counts_5_intron = np.zeros(4**kmer, dtype = np.dtype("i"))
    set1_counts_5_exon = np.zeros(4**kmer, dtype = np.dtype("i"))
    set1_counts_3_intron = np.zeros(4**kmer, dtype = np.dtype("i"))
    set1_counts_3_exon = np.zeros(4**kmer, dtype = np.dtype("i"))
    set2_counts_5_intron = np.zeros(4**kmer, dtype = np.dtype("i"))
    set2_counts_5_exon = np.zeros(4**kmer, dtype = np.dtype("i"))
    set2_counts_3_intron = np.zeros(4**kmer, dtype = np.dtype("i"))
    set2_counts_3_exon = np.zeros(4**kmer, dtype = np.dtype("i"))
    
    for gene in geneNames:
        set1Threes = np.nonzero(set1[gene] == B3)[0]
        set1Fives = np.nonzero(set1[gene] == B5)[0]
        set2Threes = np.nonzero(set2[gene] == B3)[0]
        set2Fives = np.nonzero(set2[gene] == B5)[0]
        
        for ss in set1Fives:
            sequence = str(genes[gene].seq[ss-sreEffect5_exon:ss-3].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[:ss-3].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: set1_counts_5_exon[s] += 1
            
            sequence = str(genes[gene].seq[ss+6:ss+sreEffect5_intron+1].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[ss+6:].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: set1_counts_5_intron[s] += 1
    
        for ss in set1Threes:
            sequence = str(genes[gene].seq[ss+4:ss+sreEffect3_exon+1].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[ss+4:].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: set1_counts_3_exon[s] += 1
            
            sequence = str(genes[gene].seq[ss-sreEffect3_intron:ss-19].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[:ss-19].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: set1_counts_3_intron[s] += 1
        
        for ss in set2Fives:
            sequence = str(genes[gene].seq[ss-sreEffect5_exon:ss-3].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[:ss-3].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: set2_counts_5_exon[s] += 1
            
            sequence = str(genes[gene].seq[ss+6:ss+sreEffect5_intron+1].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[ss+6:].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: set2_counts_5_intron[s] += 1
    
        for ss in set2Threes:
            sequence = str(genes[gene].seq[ss+4:ss+sreEffect3_exon+1].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[ss+4:].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: set2_counts_3_exon[s] += 1
            
            sequence = str(genes[gene].seq[ss-sreEffect3_intron:ss-19].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[:ss-19].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: set2_counts_3_intron[s] += 1
    
    return(set1_counts_5_intron, set1_counts_5_exon, set1_counts_3_intron, set1_counts_3_exon, 
           set2_counts_5_intron, set2_counts_5_exon, set2_counts_3_intron, set2_counts_3_exon)

def get_hexamer_real_decoy_scores(geneNames, trueSeqs, decoySS, genes, kmer, sreEffect5_exon, sreEffect5_intron, sreEffect3_exon, sreEffect3_intron):
    # Get the real versus decoy scores for all hexamers
    true_counts_5_intron, true_counts_5_exon, true_counts_3_intron, true_counts_3_exon, decoy_counts_5_intron, decoy_counts_5_exon, decoy_counts_3_intron, decoy_counts_3_exon = get_hexamer_real_decoy_counts(geneNames, trueSeqs, decoySS, genes, kmer = kmer, sreEffect5_exon = sreEffect5_exon, sreEffect5_intron = sreEffect5_intron, sreEffect3_exon = sreEffect3_exon, sreEffect3_intron = sreEffect3_intron)
    
    # Add pseudocounts
    true_counts_5_intron = true_counts_5_intron + 1
    true_counts_5_exon = true_counts_5_exon + 1
    true_counts_3_intron = true_counts_3_intron + 1
    true_counts_3_exon = true_counts_3_exon + 1
    decoy_counts_5_intron = decoy_counts_5_intron + 1
    decoy_counts_5_exon = decoy_counts_5_exon + 1
    decoy_counts_3_intron = decoy_counts_3_intron + 1
    decoy_counts_3_exon = decoy_counts_3_exon + 1
    
    true_counts_intron = true_counts_5_intron + true_counts_3_intron
    true_counts_exon = true_counts_5_exon + true_counts_3_exon
    decoy_counts_intron = decoy_counts_5_intron + decoy_counts_3_intron
    decoy_counts_exon = decoy_counts_5_exon + decoy_counts_3_exon
    
    trueFreqs_intron = np.exp(np.log(true_counts_intron) - np.log(np.sum(true_counts_intron))) 
    decoyFreqs_intron = np.exp(np.log(decoy_counts_intron) - np.log(np.sum(decoy_counts_intron)))
    trueFreqs_exon = np.exp(np.log(true_counts_exon) - np.log(np.sum(true_counts_exon)))
    decoyFreqs_exon = np.exp(np.log(decoy_counts_exon) - np.log(np.sum(true_counts_exon)))
    
    sreScores_intron = np.exp(np.log(true_counts_intron) - np.log(np.sum(true_counts_intron)) 
                              - np.log(decoy_counts_intron) + np.log(np.sum(decoy_counts_intron)))
    sreScores_exon = np.exp(np.log(true_counts_exon) - np.log(np.sum(true_counts_exon)) 
                            - np.log(decoy_counts_exon) + np.log(np.sum(decoy_counts_exon)))
    
    sreScores3_intron = np.exp(np.log(true_counts_3_intron) - np.log(np.sum(true_counts_3_intron)) 
                                - np.log(decoy_counts_3_intron) + np.log(np.sum(decoy_counts_3_intron)))
    sreScores3_exon = np.exp(np.log(true_counts_3_exon) - np.log(np.sum(true_counts_3_exon)) 
                              - np.log(decoy_counts_3_exon) + np.log(np.sum(decoy_counts_3_exon)))
    
    sreScores5_intron = np.exp(np.log(true_counts_5_intron) - np.log(np.sum(true_counts_5_intron)) 
                                - np.log(decoy_counts_5_intron) + np.log(np.sum(decoy_counts_5_intron)))
    sreScores5_exon = np.exp(np.log(true_counts_5_exon) - np.log(np.sum(true_counts_5_exon)) 
                              - np.log(decoy_counts_5_exon) + np.log(np.sum(decoy_counts_5_exon)))
    
    return(sreScores_intron, sreScores_exon, sreScores3_intron, sreScores3_exon, sreScores5_intron, sreScores5_exon)
    
def score_sequences(sequences, double [:, :] exonicSREs5s, double [:, :] exonicSREs3s, double [:, :] intronicSREs5s, double [:, :] intronicSREs3s, int k = 6, int sreEffect5_exon = 80, int sreEffect5_intron = 80, int sreEffect3_exon = 80, int sreEffect3_intron = 80, meDir = ''): #/home/kmccue/projects/scfg/code/maxEnt/triplets-9mer-max/
    # Inputs:
    #  - sequences: batch_size lenght list of sequence strings, maybe need about 21 kb/base for the longest string
    #  - transitions: list of two values [pME, pEE]
    #  - pIL: numpy array of intron length distribution
    #  - pELS: decay coefficient for transcribing state eg 1/10000
    #  - pELF: numpy array of first exon length distribution
    #  - pELM: numpy array of middle exon length distribution
    #  - pELL: numpy array of last exon length distribution
    #  - delay: number of bases to delay predictions by
    #  - sreScores: sre scores for each kmer
    #  - meDir: location of the maxent model lookups
    # Outputs:
    #  - bestPath: numpy array containing the cotranscriptional viterbi best paths for each input sequence
    #  - loglik: numpy array containing the calculated log likelihoods
    #  - tbStates: numpy array containing the states at which a traceback event occurred
    #  - tblast3: numpy array containing the last 3'ss when which a traceback event occurred
    #  - emissions5: final 5' emissions
    #  - emissions3: final 3' emissions
    #  - traceback5: the 5'ss tracebacks
    #  - traceback3: the 3'ss tracebacks
    cdef int batch_size, g, L, d, i, t, ssRange
    batch_size = len(sequences)
    cdef int [:] lengths = np.zeros(batch_size, dtype=np.dtype("i"))
    
    # Collect the lengths of each sequence in the batch
    for g in range(batch_size): 
        lengths[g] = len(sequences[g])
    L = np.max(lengths.base)
    
    cdef double [:, :] emissions3 = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))
    cdef double [:, :] emissions5 = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))
    
    # Get the emissions and apply sre scores to them
    for g in range(batch_size): 
        # 5'SS exonic effects (upstream)
        ssRange = 3
        emissions5.base[g,:lengths[g]] = np.log(maxEnt5_single(sequences[g].lower(), meDir))
        emissions5.base[g,k+ssRange:lengths[g]] += np.cumsum(exonicSREs5s.base[g,:lengths[g]-k+1])[:-1-ssRange]
        emissions5.base[g,sreEffect5_exon+1:lengths[g]] -= np.cumsum(exonicSREs5s.base[g,:lengths[g]-k+1])[:-(sreEffect5_exon+1)+(k-1)]
        
        # 3'SS intronic effects (upstream)
        ssRange = 19
        emissions3.base[g,:lengths[g]] = np.log(maxEnt3_single(sequences[g].lower(), meDir))
        emissions3.base[g,k+ssRange:lengths[g]] += np.cumsum(intronicSREs3s.base[g,:lengths[g]-k+1])[:-1-ssRange]
        emissions3.base[g,sreEffect3_intron+1:lengths[g]] -= np.cumsum(intronicSREs3s.base[g,:lengths[g]-k+1])[:-(sreEffect3_intron+1)+(k-1)]
        
        # 5'SS intronic effects (downstream)
        ssRange = 4
        emissions5.base[g,:lengths[g]-sreEffect5_intron] += np.cumsum(intronicSREs5s.base[g,:lengths[g]-k+1])[sreEffect5_intron-k+1:]
        emissions5.base[g,lengths[g]-sreEffect5_intron:lengths[g]-k+1-ssRange] += np.sum(intronicSREs5s.base[g,:lengths[g]-k+1])
        emissions5.base[g,:lengths[g]-k+1-ssRange] -= np.cumsum(intronicSREs5s.base[g,ssRange:lengths[g]-k+1])
        
        # 3'SS exonic effects (downstream)
        ssRange = 3
        emissions3.base[g,:lengths[g]-sreEffect5_exon] += np.cumsum(exonicSREs3s.base[g,:lengths[g]-k+1])[sreEffect5_exon-k+1:]
        emissions3.base[g,lengths[g]-sreEffect5_exon:lengths[g]-k+1-ssRange] += np.sum(exonicSREs3s.base[g,:lengths[g]-k+1])
        emissions3.base[g,:lengths[g]-k+1-ssRange] -= np.cumsum(exonicSREs3s.base[g,ssRange:lengths[g]-k+1])
        
    return np.exp(emissions5.base), np.exp(emissions3.base)
                 
def cass_accuracy_metrics(scored_sequences_5, scored_sequences_3, geneNames, trueSeqs, B3 = 3, B5 = 5):
    # Get the best cutoff and the associated metrics for the CASS scored sequences
    true_scores = []
    for g, gene in enumerate(geneNames):
        L = len(trueSeqs[gene])
        for score in np.log2(scored_sequences_5[g,:L][trueSeqs[gene] == B5]): true_scores.append(score)
        for score in np.log2(scored_sequences_3[g,:L][trueSeqs[gene] == B3]): true_scores.append(score)
    min_score = np.min(true_scores)
    if np.isnan(min_score): 
        return 0, 0, 0, min_score
    
    all_scores = []
    for g, gene in enumerate(geneNames):
        L = len(trueSeqs[gene])
        for score in np.log2(scored_sequences_5[g,:L][trueSeqs[gene] != B5]): 
            if score > min_score: all_scores.append(score)
        for score in np.log2(scored_sequences_3[g,:L][trueSeqs[gene] != B3]): 
            if score > min_score: all_scores.append(score)
    
    all_scores = np.array(true_scores + all_scores)
    all_scores_bool = np.zeros(len(all_scores), dtype=np.dtype("i"))
    all_scores_bool[:len(true_scores)] = 1
    sort_inds = np.argsort(all_scores)
    all_scores = all_scores[sort_inds]
    all_scores_bool = all_scores_bool[sort_inds]
    
    num_all_positives = len(true_scores)
    num_all = len(all_scores)
    best_f1 = 0
    best_cutoff = 0
    for i, cutoff in enumerate(all_scores):
        if all_scores_bool[i] == 0: continue
        true_positives = np.sum(all_scores_bool[i:])
        false_negatives = num_all_positives - true_positives
        false_positives = num_all - i - true_positives
        
        ssSens = true_positives / (true_positives + false_negatives)
        ssPrec = true_positives / (true_positives + false_positives)
        f1 = 2 / (1/ssSens + 1/ssPrec)
        if f1 >= best_f1:
            best_f1 = f1
            best_cutoff = cutoff
            best_sens = ssSens
            best_prec = ssPrec
        
    return best_sens, best_prec, best_f1, best_cutoff
    
def cass_accuracy_metrics_set_cutoff(scored_sequences_5, scored_sequences_3, geneNames, trueSeqs, cutoff, B3 = 3, B5 = 5):
    # Get the associated metrics for the CASS scored sequences with a given cutoff
    true_scores = []
    for g, gene in enumerate(geneNames):
        L = len(trueSeqs[gene])
        for score in np.log2(scored_sequences_5[g,:L][trueSeqs[gene] == B5]): true_scores.append(score)
        for score in np.log2(scored_sequences_3[g,:L][trueSeqs[gene] == B3]): true_scores.append(score)
    min_score = np.min(true_scores)
    if np.isnan(min_score): 
        return 0, 0, 0
    
    all_scores = []
    for g, gene in enumerate(geneNames):
        L = len(trueSeqs[gene])
        for score in np.log2(scored_sequences_5[g,:L][trueSeqs[gene] != B5]): 
            if score > min_score: all_scores.append(score)
        for score in np.log2(scored_sequences_3[g,:L][trueSeqs[gene] != B3]): 
            if score > min_score: all_scores.append(score)
    
    all_scores = np.array(true_scores + all_scores)
    all_scores_bool = np.zeros(len(all_scores), dtype=np.dtype("i"))
    all_scores_bool[:len(true_scores)] = 1
    sort_inds = np.argsort(all_scores)
    all_scores = all_scores[sort_inds]
    all_scores_bool = all_scores_bool[sort_inds]
    
    num_all_positives = len(true_scores)
    num_all = len(all_scores)
    
    true_positives = np.sum((all_scores > cutoff)&(all_scores_bool == 1))
    false_negatives = num_all_positives - true_positives
    false_positives = np.sum((all_scores > cutoff)&(all_scores_bool == 0))
    
    ssSens = true_positives / (true_positives + false_negatives)
    ssPrec = true_positives / (true_positives + false_positives)
    f1 = 2 / (1/ssSens + 1/ssPrec)
        
    return ssSens, ssPrec, f1

def order_genes(geneNames, genes, num_threads):
    # Re-order genes to feed into parallelized prediction algorithm to use parallelization efficiently
    # geneNames: list of names of genes to re-order based on length 
    # num_threads: number of threads available to parallelize across
    lengthsOfGenes = np.array([len(str(genes[gene].seq)) for gene in geneNames])
    geneNames = geneNames[np.argsort(lengthsOfGenes)]
    geneNames = np.flip(geneNames)

    # ordering the genes for optimal processing
    l = len(geneNames)
    ind = l - l//num_threads
    longest_thread = []
    for i in np.flip(range(num_threads)):
        longest_thread.append(ind)
        ind -= (l//num_threads + int(i<=(l%num_threads)))
    
    indices = longest_thread.copy()
    for i in range(1,l//num_threads):
        indices += list(np.array(longest_thread) + i)
    
    ind = l//num_threads
    for i in range(l%num_threads): indices.append(ind + i*l%num_threads)

    indices = np.argsort(indices)
    return(geneNames[indices])
    
def viterbi(sequences, transitions, double [:] pIL, double [:] pELS, double [:] pELF, double [:] pELM, double [:] pELL, double [:, :] exonicSREs5s, double [:, :] exonicSREs3s, double [:, :] intronicSREs5s, double [:, :] intronicSREs3s, int k, int sreEffect5_exon = 80, int sreEffect5_intron = 80, int sreEffect3_exon = 80, int sreEffect3_intron = 80, meDir = ''): #/home/kmccue/projects/scfg/code/maxEnt/triplets-9mer-max/
    # Inputs:
    #  - sequences: batch_size lenght list of sequence strings, maybe need about 21 kb/base for the longest string
    #  - transitions: list of two values [pME, pEE]
    #  - pIL: numpy array of intron length distribution
    #  - pELS: decay coefficient for transcribing state eg 1/10000
    #  - pELF: numpy array of first exon length distribution
    #  - pELM: numpy array of middle exon length distribution
    #  - pELL: numpy array of last exon length distribution
    #  - delay: number of bases to delay predictions by
    #  - sreScores: sre scores for each kmer
    #  - meDir: location of the maxent model lookups
    # Outputs:
    #  - bestPath: numpy array containing the cotranscriptional viterbi best paths for each input sequence
    #  - loglik: numpy array containing the calculated log likelihoods
    #  - tbStates: numpy array containing the states at which a traceback event occurred
    #  - tblast3: numpy array containing the last 3'ss when which a traceback event occurred
    #  - emissions5: final 5' emissions
    #  - emissions3: final 3' emissions
    #  - traceback5: the 5'ss tracebacks
    #  - traceback3: the 3'ss tracebacks
    cdef int batch_size, g, L, d, i, t, ssRange
    cdef double pME, p1E, pEE, pEO
    
    batch_size = len(sequences)
    
#     cdef int [:] t = np.zeros(batch_size, dtype=np.dtype("i"))
    cdef int [:] tbindex = np.zeros(batch_size, dtype=np.dtype("i"))
    cdef int [:] lengths = np.zeros(batch_size, dtype=np.dtype("i"))
    cdef double [:] loglik = np.log(np.zeros(batch_size, dtype=np.dtype("d")))
    
    # Collect the lengths of each sequence in the batch
    for g in range(batch_size): 
        lengths[g] = len(sequences[g])
    L = np.max(lengths.base)
    
    cdef double [:, :] emissions3 = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))
    cdef double [:, :] emissions5 = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))
     
    cdef double [:, :] Three = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))    
    cdef double [:, :] Five = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))
    cdef int [:, :] traceback5 = np.zeros((batch_size, L), dtype=np.dtype("i")) + L
    cdef int [:, :] traceback3 = np.zeros((batch_size, L), dtype=np.dtype("i")) + L
    cdef int [:, :] bestPath = np.zeros((batch_size, L), dtype=np.dtype("i"))
    
    # Rewind state vars
    cdef int exon = 2
    cdef int intron = 1
     
    # Convert inputs to log space
    transitions = np.log(transitions)
    pIL = np.log(pIL)
    pELS = np.log(pELS)
    pELF = np.log(pELF)
    pELM = np.log(pELM)
    pELL = np.log(pELL)
    
    # Get the emissions and apply sre scores to them
    for g in range(batch_size): 
        # 5'SS exonic effects (upstream)
        ssRange = 3
        emissions5.base[g,:lengths[g]] = np.log(maxEnt5_single(sequences[g].lower(), meDir))
        emissions5.base[g,k+ssRange:lengths[g]] += np.cumsum(exonicSREs5s.base[g,:lengths[g]-k+1])[:-1-ssRange]
        emissions5.base[g,sreEffect5_exon+1:lengths[g]] -= np.cumsum(exonicSREs5s.base[g,:lengths[g]-k+1])[:-(sreEffect5_exon+1)+(k-1)]
        
        # 3'SS intronic effects (upstream)
        ssRange = 19
        emissions3.base[g,:lengths[g]] = np.log(maxEnt3_single(sequences[g].lower(), meDir))
        emissions3.base[g,k+ssRange:lengths[g]] += np.cumsum(intronicSREs3s.base[g,:lengths[g]-k+1])[:-1-ssRange]
        emissions3.base[g,sreEffect3_intron+1:lengths[g]] -= np.cumsum(intronicSREs3s.base[g,:lengths[g]-k+1])[:-(sreEffect3_intron+1)+(k-1)]
        
        # 5'SS intronic effects (downstream)
        ssRange = 4
        emissions5.base[g,:lengths[g]-sreEffect5_intron] += np.cumsum(intronicSREs5s.base[g,:lengths[g]-k+1])[sreEffect5_intron-k+1:]
        emissions5.base[g,lengths[g]-sreEffect5_intron:lengths[g]-k+1-ssRange] += np.sum(intronicSREs5s.base[g,:lengths[g]-k+1])
        emissions5.base[g,:lengths[g]-k+1-ssRange] -= np.cumsum(intronicSREs5s.base[g,ssRange:lengths[g]-k+1])
        
        # 3'SS exonic effects (downstream)
        ssRange = 3
        emissions3.base[g,:lengths[g]-sreEffect5_exon] += np.cumsum(exonicSREs3s.base[g,:lengths[g]-k+1])[sreEffect5_exon-k+1:]
        emissions3.base[g,lengths[g]-sreEffect5_exon:lengths[g]-k+1-ssRange] += np.sum(exonicSREs3s.base[g,:lengths[g]-k+1])
        emissions3.base[g,:lengths[g]-k+1-ssRange] -= np.cumsum(exonicSREs3s.base[g,ssRange:lengths[g]-k+1])
    
    # Convert the transition vector into named probabilities
    pME = transitions[0]
    p1E = np.log(1 - np.exp(pME))
    pEE = transitions[1]
    pEO = np.log(1 - np.exp(pEE))
    
    # Initialize the first and single exon probabilities
    cdef double [:] ES = np.zeros(batch_size, dtype=np.dtype("d"))
    for g in range(batch_size): ES[g] = pELS[L-1] + p1E
    
    for g in prange(batch_size, nogil=True): # parallelize over the sequences in the batch
        for t in range(1,lengths[g]):
            Five[g,t] = pELF[t-1]
            
            for d in range(t,0,-1):
                # 5'SS
                if pEE + Three[g,t-d-1] + pELM[d-1] > Five[g,t]:
                    traceback5[g,t] = d
                    Five[g,t] = pEE + Three[g,t-d-1] + pELM[d-1]
            
                # 3'SS
                if Five[g,t-d-1] + pIL[d-1] > Three[g,t]:
                    traceback3[g,t] = d
                    Three[g,t] = Five[g,t-d-1] + pIL[d-1]
                    
            Five[g,t] += emissions5[g,t]
            Three[g,t] += emissions3[g,t]
            
        # TODO: Add back in single exon case for if we ever come back to that
        for i in range(1, lengths[g]):
            if pME + Three[g,i] + pEO + pELL[lengths[g]-i-2] > loglik[g]:
                loglik[g] = pME + Three[g,i] + pEO + pELL[lengths[g]-i-2]
                tbindex[g] = i
                
        if ES[g] <= loglik[g]: # If the single exon case isn't better, trace back
            while 0 < tbindex[g]:
                bestPath[g,tbindex[g]] = 3
                tbindex[g] -= traceback3[g,tbindex[g]] + 1
                bestPath[g,tbindex[g]] = 5
                tbindex[g] -= traceback5[g,tbindex[g]] + 1 
        else:
            loglik[g] = ES[g]


    return bestPath.base, loglik.base, emissions5.base, emissions3.base





def emissions(sequences, exonicSREs5s, exonicSREs3s, intronicSREs5s, intronicSREs3s, k, sreEffect5_exon, sreEffect5_intron, sreEffect3_exon, sreEffect3_intron, meDir = ''): 
    # Get the best parses of all the input sequences
    
    batch_size = len(sequences)
    lengths = np.zeros(batch_size, dtype=np.dtype("i"))
    
    # Collect the lengths of each sequence in the batch
    for g in range(batch_size): 
        lengths[g] = len(sequences[g])
    L = np.max(lengths)
    
    emissions3 = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))
    emissions5 = np.log(np.zeros((batch_size, L), dtype=np.dtype("d"))) 

    # Get the emissions and apply sre scores to them
    for g in range(batch_size): 
        # 5'SS exonic effects (upstream)
        ssRange = 3
        emissions5[g,:lengths[g]] = np.log(maxEnt5_single(sequences[g].lower(), meDir))
        emissions5[g,k+ssRange:lengths[g]] += np.cumsum(exonicSREs5s[g,:lengths[g]-k+1])[:-1-ssRange]
        emissions5[g,sreEffect5_exon+1:lengths[g]] -= np.cumsum(exonicSREs5s[g,:lengths[g]-k+1])[:-(sreEffect5_exon+1)+(k-1)]
        
        # 3'SS intronic effects (upstream)
        ssRange = 19
        emissions3[g,:lengths[g]] = np.log(maxEnt3_single(sequences[g].lower(), meDir))
        emissions3[g,k+ssRange:lengths[g]] += np.cumsum(intronicSREs3s[g,:lengths[g]-k+1])[:-1-ssRange]
        emissions3[g,sreEffect3_intron+1:lengths[g]] -= np.cumsum(intronicSREs3s[g,:lengths[g]-k+1])[:-(sreEffect3_intron+1)+(k-1)]
        
        # 5'SS intronic effects (downstream)
        ssRange = 4
        emissions5[g,:lengths[g]-sreEffect5_intron] += np.cumsum(intronicSREs5s[g,:lengths[g]-k+1])[sreEffect5_intron-k+1:]
        emissions5[g,lengths[g]-sreEffect5_intron:lengths[g]-k+1-ssRange] += np.sum(intronicSREs5s[g,:lengths[g]-k+1])
        emissions5[g,:lengths[g]-k+1-ssRange] -= np.cumsum(intronicSREs5s[g,ssRange:lengths[g]-k+1])
        
        # 3'SS exonic effects (downstream)
        ssRange = 3
        emissions3[g,:lengths[g]-sreEffect5_exon] += np.cumsum(exonicSREs3s[g,:lengths[g]-k+1])[sreEffect5_exon-k+1:]
        emissions3[g,lengths[g]-sreEffect5_exon:lengths[g]-k+1-ssRange] += np.sum(exonicSREs3s[g,:lengths[g]-k+1])
        emissions3[g,:lengths[g]-k+1-ssRange] -= np.cumsum(exonicSREs3s[g,ssRange:lengths[g]-k+1])

        
    return emissions5, emissions3



def viterbi_FBP(sequences, double[:, :] emissions5, double[:, :] emissions3, double pME, double pEE, double pEO, double [:] pIL, double [:] pELF, double [:] pELM, double [:] pELL, ): #/home/kmccue/projects/scfg/code/maxEnt/triplets-9mer-max/

    cdef int batch_size, g, L, d, i, t, ssRange
    
    batch_size = len(sequences)
    
    # cdef int [:] t = np.zeros(batch_size, dtype=np.dtype("i"))
    cdef int [:] tbindex = np.zeros(batch_size, dtype=np.dtype("i"))
    cdef int [:] lengths = np.zeros(batch_size, dtype=np.dtype("i"))
    cdef double [:] loglik = np.log(np.zeros(batch_size, dtype=np.dtype("d")))
    
    # Collect the lengths of each sequence in the batch
    for g in range(batch_size): 
        lengths[g] = len(sequences[g])
    L = np.max(lengths.base)
    

     
    cdef double [:, :] Three = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))    
    cdef double [:, :] Five = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))
    cdef int [:, :] traceback5 = np.zeros((batch_size, L), dtype=np.dtype("i")) + L
    cdef int [:, :] traceback3 = np.zeros((batch_size, L), dtype=np.dtype("i")) + L
    cdef int [:, :] bestPath = np.zeros((batch_size, L), dtype=np.dtype("i"))
    
    # Rewind state vars
    cdef int exon = 2
    cdef int intron = 1
    
    
    for g in prange(batch_size, nogil=True): # parallelize over the sequences in the batch
        for t in range(1,lengths[g]):
            Five[g,t] = pELF[t-1]
            
            for d in range(t,0,-1):
                # 5'SS
                if pEE + Three[g,t-d-1] + pELM[d-1] > Five[g,t]:
                    traceback5[g,t] = d
                    Five[g,t] = pEE + Three[g,t-d-1] + pELM[d-1]
            
                # 3'SS
                if Five[g,t-d-1] + pIL[d-1] > Three[g,t]:
                    traceback3[g,t] = d
                    Three[g,t] = Five[g,t-d-1] + pIL[d-1]
                    
            Five[g,t] += emissions5[g, t]
            Three[g,t] += emissions3[g, t]
            
        # TODO: Add back in single exon case for if we ever come back to that
        for i in range(1, lengths[g]):
            if pME + Three[g,i] + pEO + pELL[lengths[g]-i-2] > loglik[g]:
                loglik[g] = pME + Three[g,i] + pEO + pELL[lengths[g]-i-2]
                tbindex[g] = i
                

        while 0 < tbindex[g]:
            bestPath[g,tbindex[g]] = 3
            tbindex[g] -= traceback3[g,tbindex[g]] + 1
            bestPath[g,tbindex[g]] = 5
            tbindex[g] -= traceback5[g,tbindex[g]] + 1 
        #else:
        #    loglik[g] = ES[g]


    return bestPath.base, loglik.base, emissions5.base, emissions3.base
