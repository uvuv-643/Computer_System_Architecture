:intr intr_enter
    10 read
    dup 10 = if 1 stop_input ! then
    11 omit
ei ;
variable stop_input
0 stop_input !
begin stop_input @ until