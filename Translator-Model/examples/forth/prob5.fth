  variable answer_string 15 allot

  :  gcd dup 0 =
      if
          drop
      else
          swap over mod gcd
      then ;

  1
  21 1 do dup i * swap dup i gcd swap drop / loop

  answer_string
  14 +
  over 10 mod
  over !
  begin
      swap 10 / swap over dup 0 = if 1 else
      swap 1 - swap 10 mod over ! 0
      then
  until

  swap
  answer_string 15 + swap do i @ 48 + 11 omit loop