import json

from datasets import load_dataset

ds = load_dataset("HuggingFaceH4/ultrachat_200k")
victims = list(ds["train_sft"]["prompt"])

with open("ultrachat_victims_5.json", "w") as f:
    json.dump({"victims": victims[:5]}, f, indent=2)

with open("ultrachat_victims_10_1.json", "w") as f:
    json.dump({"victims": victims[0:10]}, f, indent=2)

with open("ultrachat_victims_10_2.json", "w") as f:
    json.dump({"victims": victims[10:20]}, f, indent=2)

with open("ultrachat_victims_10_3.json", "w") as f:
    json.dump({"victims": victims[20:30]}, f, indent=2)

with open("ultrachat_victims_10_4.json", "w") as f:
    json.dump({"victims": victims[30:40]}, f, indent=2)

with open("ultrachat_victims_10_5.json", "w") as f:
    json.dump({"victims": victims[40:50]}, f, indent=2)

with open("ultrachat_victims_10_6.json", "w") as f:
    json.dump({"victims": victims[50:60]}, f, indent=2)

with open("ultrachat_victims_10_7.json", "w") as f:
    json.dump({"victims": victims[60:70]}, f, indent=2)

with open("ultrachat_victims_10_8.json", "w") as f:
    json.dump({"victims": victims[70:80]}, f, indent=2)

with open("ultrachat_victims_10_9.json", "w") as f:
    json.dump({"victims": victims[80:90]}, f, indent=2)

with open("ultrachat_victims_10_10.json", "w") as f:
    json.dump({"victims": victims[90:100]}, f, indent=2)

with open("ultrachat_victims_100.json", "w") as f:
    json.dump({"victims": victims[:100]}, f, indent=2)

with open("ultrachat_victims_500.json", "w") as f:
    json.dump({"victims": victims[:500]}, f, indent=2)
