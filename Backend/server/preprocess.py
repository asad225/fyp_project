# import pandas as pd
# data = pd.read_csv("BankFAQs.csv")
# import json
# intents = [ {
#       "tag": list(row)[0][2],
#       "patterns": [list(row)[0][0]],
#       "responses": [list(row)[0][1]]
#     }for row in zip(data.values)]

# dict = {'intents':intents}
# json.dump(dict,open("intents.json","w"),indent=5)

import pandas as pd
import json

data = pd.read_csv("mental_health.csv")

def create_tag(pattern):
    tag = pattern.replace(" ", "_")  # Replace spaces with underscores
    return tag.lower()

intents = [{
    "tag": create_tag(row[0]),       # Use the modified pattern sentence as the tag
    "patterns": [row[0]],            # Use the original pattern sentence as the pattern
    "responses": [row[1]]
} for row in data.values]

dict_data = {'intents': intents}

# Save the JSON data to a file
with open("mental_health_intents.json", "w") as json_file:
    json.dump(dict_data, json_file, indent=4)
