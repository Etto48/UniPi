def test(archive, c, m):
    if archive.not_full():
        archive.add(m)
    elif archive.has_more_crowded_than(m):
        archive.replace_worst(m)
    if archive.compare_crowdedness(c, m): # c is more crowded than m
        return m
    else:
        return c