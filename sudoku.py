def sudoku(f):
  def cp(q, s):
    l = set(s[q[0]])
    l |= {s[i][q[1]] for i in range(9)}
    k = q[0] // 3, q[1] // 3
    for i in range(3):
      l |= set(s[k[0] * 3 + i][k[1] * 3 : (k[1] + 1) * 3])
    return set(range(1, 10)) - l

  def ec(l):
    q = set(l) - {0}
    for c in q:
      if l.count(c) != 1:
        return True
    return False

  s, t = [], []

  for nl, l in enumerate(f):
    try: n = list(map(int, l))
    except: return
    if len(n) != 9: return
    t += [[nl, i] for i in range(9) if n[i] == 0]
    s.append(n)
  
  if nl != 8: return

  for l in range(9):
    if ec(s[l]): return

  for c in range(9):
    k = [s[l][c] for l in range(9)]
    if ec(k): return

  for l in range(3):
    for c in range(3):
      q = []
      for i in range(3):
        q += s[l * 3 + i][c * 3 : (c + 1) * 3]
      if ec(q): return

  p, cr = [[] for i in t], 0

  while cr < len(t):
    p[cr] = cp(t[cr], s)
    
    try:
      while not p[cr]:
        s[t[cr][0]][t[cr][1]] = 0
        cr -= 1
    except: return
    
    s[t[cr][0]][t[cr][1]] = p[cr].pop()
    cr += 1

  return s
