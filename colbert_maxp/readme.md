# ColBERT MaxP 

#### Process 
Use sliding window of 225 tokens, stride of 200. 

Query is split into n pieces: Q = {q1, ..., qn} 

Document is split into m pieces: D = {d1, ..., dm} 

We calculate the score as follows: 
S_q1 = Max(colbert(q1,d1),...,colbert(q1,dm)) 

S_q2 = Max(colbert(q2,d1),...,colbert(q2,dm)) 

...

S_qn = Max(colbert(qn,d1),...,colbert(qn,dm)) 

The final score is obtained as follows: 

S = Max(S_q1, S_q2, ..., S_qn) 
