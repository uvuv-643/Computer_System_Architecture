variable stop_input
variable str_buff 40 allot
variable str_len

0 str_len !
0 stop_input !

11 ." hello! enter your name: "
begin stop_input @ until
11 ." hello, "
str_len @ 0 do str_buff i + @ 11 omit loop
11 ." !"

:intr intr_enter di
    10 read
    dup 13 = if 1 stop_input ! else
        str_buff str_len @ + !
        str_len @ 1 + str_len !
    then
ei ;
