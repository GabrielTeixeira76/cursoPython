def calcular_sensibilidade(VP, FN):
    return VP / (VP + FN)

def calcular_especificidade(VN, FP):
    return VN / (FP + VN)

def calcular_precisao(VP, FP):
    return VP / (VP + FP)

def calcular_acuracia(VP, VN, FP, FN):
    return (VP + VN) / (VP + VN + FP + FN)

def calcular_fscore(precisao, sensibilidade):
    return 2 * (precisao * sensibilidade) / (precisao + sensibilidade)