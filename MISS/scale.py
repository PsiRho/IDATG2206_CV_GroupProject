
# Great success
def scale(old):
    absW = 0.1 * old
    edgeW = 0.3 * old
    kernelW = 0.6 * old
    return (absW+edgeW+kernelW)/3

