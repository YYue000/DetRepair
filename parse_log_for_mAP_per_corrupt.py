
results = []

with open('log') as fr:
    r = {}
    for line in fr.readlines():
        if line.startswith('Testing') and 'at severity' in line:
            sp = line.split(' ')
            r['corrupt'] = sp[1]
            r['severity'] = int(sp[-1])
            continue
        if 'Average Precision' in line and ' all' in line and ':' in line:
            r['ap'] = float(line.split(' ')[-1])
            results.append(r)
            r = {}

s = '|---corruption\severity---|'
Severity = 5
for _ in range(Severity):
    s+=f'----{_+1}----|'
s+='\n'

Corrupts = len(results)//Severity
for c in range(Corrupts):
    r = results[c*Severity]
    s+=f'|{r["corrupt"]}|'
    for _ in range(Severity):
        r = results[c*Severity+_]
        s+=f'{r["ap"]}|'
        #s+=f'{r["ap"]}/{r["corrupt"]}/{r["severity"]}|'
    s+='\n'
print(s)

print(sum([_["ap"] for _ in results])/len(results))

                
