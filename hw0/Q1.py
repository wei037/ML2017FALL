import sys

text = sys.argv[1]

with open(text) as f :
	data = f.read()
f.close()
data = data.replace('\n',' ')
data_list = data.split(' ')[:-1]

word = []
cnt = []
for i in range(len(data_list)) :
	found = 0
	for j in range(len(word)) :
		if data_list[i] == word[j] :
			found = 1
			cnt[j] += 1
			break
	if found == 0 :
		word.append(data_list[i])
		cnt.append(1)

out = open('Q1.txt', 'w')

for i in range(len(word)) :
	str1 = str(word[i]) + ' ' + str(i) + ' ' + str(cnt[i])
	if i != len(word) - 1 :
		str1 = str1 + '\n'
	out.write(str1)	

out.close()
