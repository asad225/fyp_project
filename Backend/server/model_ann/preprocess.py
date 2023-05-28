import pandas as pd
data = pd.read_csv("BankFAQs.csv")
import json
intents = [ {
      "tag": list(row)[0][2],
      "patterns": [list(row)[0][0]],
      "responses": [list(row)[0][1]]
    }for row in zip(data.values)]

dict = {'intents':intents}
json.dump(dict,open("intents.json","w"),indent=5)