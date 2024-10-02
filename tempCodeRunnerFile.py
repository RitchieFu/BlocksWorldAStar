 = []
  while node:
    path.append(node.state)
    node = node.parent
  path.reverse()
  retur