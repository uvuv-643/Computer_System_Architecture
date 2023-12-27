:  gcd dup 0 =
    if
        drop
    else
        swap over mod gcd
    then ;

1
21 1 do dup i * swap dup i gcd swap drop / loop
11 .