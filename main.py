from utils import CopelandGalaiCalc

model = CopelandGalaiCalc()
print("------- Normal  Distribuition -------")
model.bid_ask_normal_distribuition(mu=102, sigma=7, pi=0.3)

print("----- Exponential Distribuition -----")
model.bid_ask_exponential_distribuition(lam=0.0075, pi=0.1)