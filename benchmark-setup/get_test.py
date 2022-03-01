
from imagecorruptions import get_corruption_names

TEST_SH = {'clean': 'test.sh', 'corrupt':'test_fog-5.sh'}

def _get_sh(corruption, serverity):
    if serverity <= 0:
        return TEST_SH['clean']

    if corruption == 'fog' and serverity == 5:
        return TEST_SH['corrupt']
    test_cs = f'test_scripts/test_{corruption}-{serverity}.sh'
    fw = open(test_cs, 'w')
    with open(TEST_SH['corrupt']) as fr:
        for line in fr.readlines():
            if 'fog' in line:
                line = line.replace('fog', corruption)
            elif 'severities' in line:
                line = line.replace('5', str(serverity))

            fw.write(line)
    fw.close()
    return test_cs
    
    

for corruption in get_corruption_names():
    for serverity in range(1, 6):
        _get_sh(corruption, serverity)


