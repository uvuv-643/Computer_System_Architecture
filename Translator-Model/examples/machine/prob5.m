0 jmp (const, 1)
1 jmp (const, 13)
2 dup
3 push (const, 0)
4 eq
5 zjmp (const, 8)
6 drop
7 jmp (const, 12)
8 swap
9 over
10 mod
11 call (const, 2)
12 ret
13 push (const, 1)
14 push (const, 21)
15 push (const, 1)
16 pop
17 pop
18 dup
19 rpop
20 dup
21 rpop
22 pop
23 pop
24 mul
25 swap
26 dup
27 rpop
28 dup
29 rpop
30 pop
31 pop
32 call (const, 2)
33 swap
34 drop
35 div
36 rpop
37 rpop
38 push (const, 1)
39 add
40 over
41 over
42 gr
43 zjmp (const, 16)
44 drop
45 drop
46 push (const, 11)
47 write
48 halt