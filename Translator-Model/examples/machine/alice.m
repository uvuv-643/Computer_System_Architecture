0 jmp (const, 25)
1 di
2 push (const, 10)
3 read
4 dup
5 push (const, 13)
6 eq
7 zjmp (const, 25)
8 push (const, 1)
9 push (const, 512)
10 store
11 jmp (const, 29)
12 push (const, 513)
13 push (const, 554)
14 load
15 add
16 store
17 push (const, 554)
18 load
19 push (const, 1)
20 add
21 push (const, 554)
22 store
23 ei
24 ret
25 push (const, 0)
26 push (const, 554)
27 store
28 push (const, 0)
29 push (const, 512)
30 store
31 push (const, 24)
32 push (const, 0)
33 store
34 push (const, 'h')
35 push (const, 1)
36 store
37 push (const, 'e')
38 push (const, 2)
39 store
40 push (const, 'l')
41 push (const, 3)
42 store
43 push (const, 'l')
44 push (const, 4)
45 store
46 push (const, 'o')
47 push (const, 5)
48 store
49 push (const, '!')
50 push (const, 6)
51 store
52 push (const, ' ')
53 push (const, 7)
54 store
55 push (const, 'e')
56 push (const, 8)
57 store
58 push (const, 'n')
59 push (const, 9)
60 store
61 push (const, 't')
62 push (const, 10)
63 store
64 push (const, 'e')
65 push (const, 11)
66 store
67 push (const, 'r')
68 push (const, 12)
69 store
70 push (const, ' ')
71 push (const, 13)
72 store
73 push (const, 'y')
74 push (const, 14)
75 store
76 push (const, 'o')
77 push (const, 15)
78 store
79 push (const, 'u')
80 push (const, 16)
81 store
82 push (const, 'r')
83 push (const, 17)
84 store
85 push (const, ' ')
86 push (const, 18)
87 store
88 push (const, 'n')
89 push (const, 19)
90 store
91 push (const, 'a')
92 push (const, 20)
93 store
94 push (const, 'm')
95 push (const, 21)
96 store
97 push (const, 'e')
98 push (const, 22)
99 store
100 push (const, ':')
101 push (const, 23)
102 store
103 push (const, ' ')
104 push (const, 24)
105 store
106 push (const, 0)
107 load
108 push (const, 0)
109 push (const, 1)
110 add
111 over
112 zjmp (const, 118)
113 load
114 omit
115 push (const, 1)
116 sub
117 jmp (const, 109)
118 push (const, 512)
119 load
120 zjmp (const, 118)
121 push (const, 7)
122 push (const, 25)
123 store
124 push (const, 'h')
125 push (const, 26)
126 store
127 push (const, 'e')
128 push (const, 27)
129 store
130 push (const, 'l')
131 push (const, 28)
132 store
133 push (const, 'l')
134 push (const, 29)
135 store
136 push (const, 'o')
137 push (const, 30)
138 store
139 push (const, ',')
140 push (const, 31)
141 store
142 push (const, ' ')
143 push (const, 32)
144 store
145 push (const, 25)
146 load
147 push (const, 25)
148 push (const, 1)
149 add
150 over
151 zjmp (const, 157)
152 load
153 omit
154 push (const, 1)
155 sub
156 jmp (const, 148)
157 push (const, 554)
158 load
159 push (const, 0)
160 pop
161 pop
162 push (const, 513)
163 rpop
164 dup
165 rpop
166 pop
167 pop
168 add
169 load
170 push (const, 11)
171 omit
172 rpop
173 rpop
174 push (const, 1)
175 add
176 over
177 over
178 gr
179 zjmp (const, 160)
180 drop
181 drop
182 push (const, 1)
183 push (const, 33)
184 store
185 push (const, '!')
186 push (const, 34)
187 store
188 push (const, 33)
189 load
190 push (const, 33)
191 push (const, 1)
192 add
193 over
194 zjmp (const, 200)
195 load
196 omit
197 push (const, 1)
198 sub
199 jmp (const, 191)
200 halt