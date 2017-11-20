import os
this_file = os.path.abspath(__file__)
print(this_file)
model_path = os.path.join(this_file, "..", "outputs", "animals.model")

print(model_path)

print(os.path.abspath(model_path))