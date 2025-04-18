import gin

@gin.configurable
def get_str(obj):
    return obj.__str__