import json


f = open("example_submission.json", encoding='utf-8')
result = json.load(f)
# print(result)
print(type(result))
print(len(result['results']))
print(type(result['results']))
