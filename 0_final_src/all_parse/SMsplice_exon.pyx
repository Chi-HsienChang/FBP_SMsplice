# SMsplice.pyx
#distutils: extra_link_args=-fopenmp
from cython import parallel
from cython.parallel import prange
import numpy as np
import time
from math import exp
from libc.math cimport exp as c_exp
cimport openmp

# 設定 NumPy 隨機種子
np.random.seed(0)
# cython: language_level=3

import math
cimport numpy as np

###############################################
# 1. 保留前 k 大的 state
###############################################
def keep_top_k_states(state_dict, k):
    """
    只保留 state_dict 中對數機率最高的前 k 個 state。
    state_dict: { state: log_alpha }
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

###############################################
# 2. 數值穩定的 logsumexp
###############################################
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

###############################################
# 3. transition_dp: 狀態轉移函數
###############################################
cdef tuple transition_dp(
    tuple state, double log_score, int pos, int symbol,
    object sequences, int length,
    double pME, double[:] pELF, double[:] pIL,
    double pEE, double[:] pELM,
    double[:] emissions5, double[:] emissions3
):
    """
    state = (used5, used3, lastSymbol, zeroCount, last5Pos, last3Pos)
    symbol: 0 (不轉移), 5 (5'SS), 3 (3'SS)
    """
    cdef:
        int used5       = state[0]
        int used3       = state[1]
        int lastSymbol  = state[2]
        int zeroCount   = state[3]
        int last5Pos    = state[4]
        int last3Pos    = state[5]
        double new_log_score = log_score
        int newUsed5 = used5
        int newUsed3 = used3
        int newZeroCount = zeroCount
        int newLast5Pos = last5Pos
        int newLast3Pos = last3Pos
        int gap_5, gap_3
        int newLastSymbol

    # symbol == 0: 不轉移，累積距離
    if symbol == 0:
        if lastSymbol == 5 or lastSymbol == 3:
            newZeroCount = zeroCount + 1
        newLastSymbol = 0

    # symbol == 5: 5'SS 事件
    elif symbol == 5:
        if emissions5[pos] <= float('-inf'):
            return None
        if pos + 1 >= length:
            return None
        if not (sequences[pos] == 'G' and sequences[pos+1] == 'T'):
            return None
        # 僅允許當前狀態無 pending 5'SS，即 used5 == used3
        if lastSymbol == 5 or (((lastSymbol == 5) or (lastSymbol == 3)) and zeroCount < float('inf')) or (used5 != used3):
            return None
        if used5 == 0:
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

    # symbol == 3: 3'SS 事件
    elif symbol == 3:
        if emissions3[pos] <= float('-inf'):
            return None
        if pos - 1 < 0:
            return None
        if not (sequences[pos] == 'G' and sequences[pos-1] == 'A'):
            return None
        # 必須已有 5'SS，並且僅允許一次 3'SS，滿足 used5 == used3 + 1
        if lastSymbol == 3 or (((lastSymbol == 5) or (lastSymbol == 3)) and zeroCount < float('inf')) or (used5 != used3 + 1):
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

###############################################
# 4. forward_dp_step: 單步 forward
###############################################
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
    # F_curr = keep_top_k_states(F_curr, top_k)
    return F_curr

###############################################
# 5. forward_backward_low_memory_exon:
#    透過 checkpointing 進行 forward-backward 計算，
#    計算第一個 exon (first_exon_dict) 與內部 exon (exon_dict) 的後驗機率，
#    並根據閉合狀態計算 partition function (common_logZ)。
###############################################
cpdef tuple forward_backward_low_memory_exon(
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
    int top_k_input=0
):
    cdef dict checkpoints = {}
    cdef dict F_current = {}
    cdef tuple init_state = (0, 0, 0, 1, -1, -1)
    F_current[init_state] = 0.0
    checkpoints[0] = F_current.copy()
    cdef int top_k = top_k_input
    cdef int pos
    # Forward pass: 儲存 checkpoint
    for pos in range(0, length):
        F_current = forward_dp_step(
            F_current, pos, sequences, length,
            pME, pELF, pIL, pEE, pELM,
            emissions5, emissions3,
            top_k
        )
        if ((pos + 1) % checkpoint_interval == 0) or (pos + 1 == length):
            checkpoints[pos + 1] = F_current.copy()
    # 計算閉合狀態的尾部得分 B_current
    cdef dict B_current = {}
    cdef int used5, used3, lastSymbol, last3Pos, ell_index
    cdef double alpha_score, tail
    for state, alpha_score in F_current.items():
        used5 = state[0]
        used3 = state[1]
        lastSymbol = state[2]
        last3Pos = state[5]
        # 閉合狀態定義：lastSymbol == 0 且 used5 == used3 且 (used5+used3) > 0
        if lastSymbol == 0 and (used5 == used3) and ((used5 + used3) > 0):
            tail = 0.0
            ell_index = (length - last3Pos) - 2
            if ell_index > 0 and ell_index < pELL.shape[0]:
                tail = pEO + pELL[ell_index]
            B_current[state] = tail
    # 利用所有閉合狀態計算 common_logZ
    cdef list terminal_logs = []
    for state, alpha_score in F_current.items():
        if state in B_current:
            if (not math.isnan(alpha_score)) and (not math.isnan(B_current[state])):
                terminal_logs.append(alpha_score + B_current[state])
    cdef double common_logZ = float('-inf')
    if terminal_logs:
        common_logZ = logsumexp(terminal_logs)
    # Backward pass: 收集第一個 exon與內部 exon 的後驗機率
    cdef dict first_exon_dict = {}   # key = (0, b)
    cdef dict exon_dict = {}         # key = (a, b), 其中 a>=0
    cdef list ckpt_positions = sorted(checkpoints.keys())
    cdef dict B_next_segment = B_current
    cdef int i, seg_start, seg_end, seg_len, j, global_pos
    cdef list seg_F, seg_B
    cdef dict B_seg
    cdef list contributions
    cdef tuple new_state_tuple, new_state
    cdef double new_log_score, b_val, val, prob, alpha_val
    cdef list allowed_symbols
    cdef int sym, a_pos, b_pos
    for i in range(len(ckpt_positions) - 1, 0, -1):
        seg_end = ckpt_positions[i]
        seg_start = ckpt_positions[i - 1]
        seg_len = seg_end - seg_start
        # 區段 forward.
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
        # 區段 backward.
        seg_B = [None]*(seg_len+1)
        seg_B[seg_len] = B_next_segment
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
            for state, alpha_val in seg_F[j].items():
                if state in B_seg:
                    b_val = B_seg[state]
                    if (not math.isnan(alpha_val)) and (not math.isnan(b_val)):
                        val = alpha_val + b_val - common_logZ
                        if val != float('-inf') and (not math.isnan(val)):
                            prob = math.exp(val)
                            # 如果狀態的 lastSymbol==5 且沒有前一個 3'SS（state[5]<0），歸為第一個 exon，
                            # 否則歸為中間 exon。
                            if state[2] == 5:
                                if state[5] < 0:
                                    b_pos = global_pos
                                    if (0, b_pos) in first_exon_dict:
                                        first_exon_dict[(0, b_pos)] += prob
                                    else:
                                        first_exon_dict[(0, b_pos)] = prob
                                else:
                                    a_pos = state[5]
                                    b_pos = global_pos
                                    if (a_pos, b_pos) in exon_dict:
                                        exon_dict[(a_pos, b_pos)] += prob
                                    else:
                                        exon_dict[(a_pos, b_pos)] = prob
        B_next_segment = seg_B[0]
    return first_exon_dict, exon_dict, common_logZ, F_current, B_current

###############################################
# 7. forward_backward_last_exon:
#    利用前面 backward 得到的 F_current 與 B_current（閉合狀態中已包含尾部得分），
#    計算最後一個 exon 的後驗機率。
#
#    規則：僅針對有剪接 (used5+used3>0) 的狀態計算最後 exon，
#         定義最後 exon 的起始位置為 state 中的 last3Pos（若 < 0 則視為 0），
#         然後利用該狀態已包含的尾部得分 (B_current[state]) 與 alpha_score，
#         計算貢獻： exp( alpha_score + B_current[state] - common_logZ )，
#         以 (start, length-1) 為 key 將相同 exon 邊界的概率累加。
###############################################
cpdef tuple forward_backward_last_exon(
    dict F_current,   # 從 forward_backward_low_memory_exon 得到的最終狀態字典
    dict B_current,   # 闭合狀態的尾部得分字典（已在 backward 過程中計算）
    int length,
    double common_logZ
):
    cdef dict last_exon_dict = {}
    cdef int used5, used3, lastSymbol, last3Pos, start
    cdef double alpha_score, tail, val, prob
    cdef tuple state

    for state, alpha_score in F_current.items():
        used5 = state[0]
        used3 = state[1]
        lastSymbol = state[2]
        last3Pos = state[5]
        # 僅計算有剪接的 parse，否則屬於 unspliced，不歸入最後 exon
        if (used5 + used3) == 0:
            continue
        # 由於在 backward 過程中，只有閉合狀態 (lastSymbol==0) 會加入尾部得分，
        # 因此我們只考慮閉合狀態
        if lastSymbol != 0:
            continue
        # 定義最後 exon 的起始位置：使用 state 中的 last3Pos (若 last3Pos < 0 則視為 0)
        start = last3Pos if last3Pos >= 0 else 0
        # 從 B_current 中取出該狀態的 tail 得分（已經計算過）
        if state in B_current:
            tail = B_current[state]
            # 狀態貢獻 = exp( alpha_score + tail - common_logZ )
            val = alpha_score + tail - common_logZ
            if val == float('-inf') or math.isnan(val):
                continue
            prob = math.exp(val)
            key = (start, 0)
            if key in last_exon_dict:
                last_exon_dict[key] += prob
            else:
                last_exon_dict[key] = prob
        else:
            tail = 0.0


    # 確保最後 exon 的結果不為空
    if not last_exon_dict:
        last_exon_dict[(0, length-1)] = 1.0

    return last_exon_dict, common_logZ
