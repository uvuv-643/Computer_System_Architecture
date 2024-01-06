0 jmp (const, 15)
1 di
2 push (const, 10)
3 read
4 dup
5 push (const, 13)
6 eq
7 zjmp (const, 15)
8 push (const, 1)
9 push (const, 512)
10 store
11 push (const, 11)
12 omit
13 ei
14 ret
15 push (const, 0)
16 push (const, 512)
17 store
18 push (const, 512)
19 load
20 zjmp (const, 18)
21 halt