import json

file_name = "json_file_name"
file_path = "./database/" + file_name

file_path_ext = file_path + ".json"

print(file_path_ext)
f = open(file_path_ext, 'r').read()

txt = ''
separation_diff = 0.9   # should be a number in (0,1)
diff_index = int(separation_diff * len(f.splitlines()))

for line in f.splitlines():
    json_dict = json.loads(line)
    txt += "__label__"
    txt += str(int(json_dict['overall']))
    txt += ' , '
    txt += json_dict['reviewText']
    txt += '\r\n'


txt_train = '\r\n'.join(txt.splitlines()[:diff_index])
txt_test = '\r\n'.join(txt.splitlines()[diff_index:])

open(file_path + ".train", 'w').write(txt_train)
open(file_path + ".test", 'w').write(txt_test)






