def get_nested_property(parent, key):
    cur_parent = parent
    key_list = key.split(".")
    for key in key_list:
        if key.isdigit():
            cur_parent = cur_parent[int(key)]
        else:
            cur_parent = getattr(cur_parent, key)
    return cur_parent
