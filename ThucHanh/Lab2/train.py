from model import FirstLM, SecondLM

first = FirstLM()
second = SecondLM()

fdata = []
r = open("1.txt", "rt", encoding="utf-8")
for s in r:
	fdata.append(s.strip())
r.close()

sdata = []
r = open("2.txt", "rt", encoding="utf-8")
for s in r:
	sdata.append(s.strip())
r.close()

first.fit(fdata)
second.fit(sdata)

first.save()
second.save()