variable stop_input

0 stop_input !

:intr intr_enter di
    10 read
    dup 13 = if 1 stop_input ! then
    11 omit
ei ;

begin stop_input @ until
