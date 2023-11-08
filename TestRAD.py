from river import anomaly

model = anomaly.ReconstructionAnomalyDetecion()

print(type(model.learn_one({"x": 0}, 1)))
print(type(model.score_one({"x": 0}, 1)))
