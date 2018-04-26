def is_numeric(n):
    if n == None:
        return False
    try:
        int(n)
        return True
    except ValueError:
        return False
