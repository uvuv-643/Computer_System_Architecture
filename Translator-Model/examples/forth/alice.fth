  :intr intr_enter
      10 read
      dup 10 = if 1 stop_input ! else
          str_buff str_len @ + !
          str_len @ 1 + str_len !
      then
  ei ;
  
  variable stop_input
  variable str_buff 40 allot
  variable str_len
  
  0 str_len !
  0 stop_input !
  
  11 ." enter your name: "
  begin stop_input @ until
  11 ." hello, "
  str_len @ 0 do str_buff i + @ 11 omit loop
  11 ." !!!"