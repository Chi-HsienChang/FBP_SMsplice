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
# 2. transition_dp: 狀態轉移
###################################################
cdef tuple transition_dp(
    tuple state, double log_score, int pos, int symbol, 
    object sequences, int length,
    double pME, double[:] pELF, double[:] pIL, 
    double pEE, double[:] pELM,
    double[:] emissions5, double[:] emissions3
):
    """
    state = (used5, used3, lastSymbol, zeroCount, last5Pos, last3Pos)
    """
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

    #------------------------------------------------
    # symbol == 0
    #------------------------------------------------
    if symbol == 0:
        # 若要確保能累加距離 => 可以令 newZeroCount = zeroCount + 1
        if lastSymbol == 5 or lastSymbol == 3:
            newZeroCount = zeroCount + 1
        newLastSymbol = 0

    #------------------------------------------------
    # symbol == 5
    #------------------------------------------------
    elif symbol == 5:
        if emissions5[pos] <= float('-inf'):
            return None
        if pos + 1 >= length:
            return None
        if not (sequences[pos] == 'G' and sequences[pos+1] == 'T'):
            return None
        # 距離或符號約束
        if lastSymbol == 5 or (
            ((lastSymbol == 5) or (lastSymbol == 3)) and zeroCount < float('inf')
        ) or (used5 != used3):
            return None

        # 第一次使用 5'SS
        if used5 == 0:
            # 注意 pos-1 可能 < 0
            if pos - 1 < 0:
                return None
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

    #------------------------------------------------
    # symbol == 3
    #------------------------------------------------
    elif symbol == 3:
        if emissions3[pos] <= float('-inf'):
            return None
        if pos - 1 < 0:
            return None
        if not (sequences[pos] == 'G' and sequences[pos-1] == 'A'):
            return None
        # 距離或符號約束
        if lastSymbol == 3 or (
            ((lastSymbol == 5) or (lastSymbol == 3)) and zeroCount < float('inf')
        ) or (used5 != used3 + 1):
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
# 3. forward_dp_step：單步 forward
###################################################
cdef dict forward_dp_step(
    dict F_prev, 
    int pos, 
    object sequences, 
    int length,
    double pME, double[:] pELF, double[:] pIL, 
    double pEE, double[:] pELM, 
    double[:] emissions5, double[:] emissions3,
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

    # 若要在每步修剪前 k 個 state，可以在此啟用
    # F_curr = keep_top_k_states(F_curr, top_k)
    return F_curr

###################################################
# 4. forward_backward_low_memory_intron(a,b)
#    只要在 backward 累加 (5'SS, 3'SS) => (a, b)
###################################################
cpdef tuple forward_backward_low_memory_intron(
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
    int checkpoint_interval,
    int top_k_input=0  # 如需可自行引入 top_k
):
    """
    與原程式結構相同（checkpointing + segment backward），
    不過在 backward 階段累加 (5'SS, 3'SS) 對應的機率，
    其中 a 為 5'SS 的位置，b 為 3'SS 的位置。

    回傳: (intron_dict, logZ)
      - intron_dict[(a, b)] = 後驗機率 (非對數)。
      - logZ = partition function 的 log 值。
    """

    cdef dict checkpoints = {}
    cdef dict F_current = {}
    cdef tuple init_state = (0, 0, 0, 1, -1, -1)
    F_current[init_state] = 0.0
    checkpoints[0] = F_current.copy()

    cdef int top_k = top_k_input  # 如果需要 top_k, 可再做調用
    cdef int pos

    #-----------------------
    # Forward pass (存 checkpoint)
    #-----------------------
    for pos in range(0, length):
        F_current = forward_dp_step(
            F_current, pos, sequences, length,
            pME, pELF, pIL, pEE, pELM,
            emissions5, emissions3,
            top_k
        )
        if ((pos + 1) % checkpoint_interval == 0) or (pos + 1 == length):
            checkpoints[pos + 1] = F_current.copy()

    #-----------------------
    # 計算 B[length]
    #-----------------------
    cdef dict B_current = {}
    cdef int used5, used3, lastSymbol, last3Pos, ell_index
    cdef double alpha_score, tail
    for state, alpha_score in F_current.items():
        used5 = state[0]
        used3 = state[1]
        lastSymbol = state[2]
        last3Pos = state[5]
        # 結尾條件
        if lastSymbol == 0 and (used5 == used3) and ((used5 + used3) > 0):
            tail = 0.0
            ell_index = (length - last3Pos) - 2
            if ell_index >= 0 and ell_index < pELL.shape[0]:
                tail += pEO + pELL[ell_index]
            B_current[state] = tail

    #-----------------------
    # partition function
    #-----------------------
    cdef list terminal_logs = []
    for state, alpha_score in F_current.items():
        if state in B_current:
            if (not math.isnan(alpha_score)) and (not math.isnan(B_current[state])):
                terminal_logs.append(alpha_score + B_current[state])
    cdef double logZ = float('-inf')
    if terminal_logs:
        logZ = logsumexp(terminal_logs)

    #-----------------------
    # intron_dict：key=(5'pos, 3'pos)，value=機率
    #-----------------------
    cdef dict intron_dict = {}

    # 從最後一個 checkpoint 往前處理
    cdef list ckpt_positions = sorted(checkpoints.keys())
    cdef dict B_next_segment = B_current

    cdef int i, seg_start, seg_end, seg_len, j, global_pos
    cdef list seg_F
    cdef list seg_B
    cdef dict B_seg
    cdef list contributions
    cdef tuple new_state_tuple, new_state
    cdef double new_log_score
    cdef double b_val, val, prob
    cdef list allowed_symbols
    cdef double alpha_val
    cdef int sym
    cdef int a_pos 
    cdef int b_pos 
    for i in range(len(ckpt_positions) - 1, 0, -1):
        seg_end = ckpt_positions[i]
        seg_start = ckpt_positions[i - 1]
        seg_len = seg_end - seg_start

        # 區段 forward
        seg_F = [None]*(seg_len+1)
        seg_F[0] = checkpoints[seg_start].copy()

        for j in range(seg_len):
            global_pos = seg_start + j
            seg_F[j+1] = forward_dp_step(
                seg_F[j], global_pos, sequences, length,
                pME, pELF, pIL, pEE, pELM,
                emissions5, emissions3,
                top_k
            )

        # 區段 backward
        seg_B = [None]*(seg_len+1)
        seg_B[seg_len] = B_next_segment

        # 自 seg_len-1 往前
        for j in range(seg_len-1, -1, -1):
            global_pos = seg_start + j
            B_seg = {}

            if global_pos == 0 or global_pos == length-1:
                allowed_symbols = [0]
            else:
                allowed_symbols = [0, 5, 3]

            for state, alpha_val in seg_F[j].items():
                if alpha_val == float('-inf') or math.isnan(alpha_val):
                    continue

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
                    if new_state in seg_B[j+1]:
                        contributions.append(new_log_score - alpha_val + seg_B[j+1][new_state])

                if contributions:
                    B_seg[state] = logsumexp(contributions)

            seg_B[j] = B_seg

            #-------------------------------
            # 累加 (a,b) = (5'SS, 3'SS)
            #-------------------------------
            for state, alpha_val in seg_F[j].items():
                if state in B_seg:
                    b_val = B_seg[state]
                    if (not math.isnan(alpha_val)) and (not math.isnan(b_val)):
                        val = alpha_val + b_val - logZ
                        if val != float('-inf') and (not math.isnan(val)):
                            prob = math.exp(val)
                            # 若 state[2] == 3 代表在 global_pos 做 3'SS，
                            # 則上一個 5'SS 的位置記錄在 state[4]
                            if state[2] == 3:
                                a_pos = state[4]
                                b_pos = global_pos
                                if a_pos >= 0:
                                    if (a_pos, b_pos) in intron_dict:
                                        intron_dict[(a_pos, b_pos)] += prob
                                    else:
                                        intron_dict[(a_pos, b_pos)] = prob

        B_next_segment = seg_B[0]

    return intron_dict, logZ
