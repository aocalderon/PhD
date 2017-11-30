#!/home/acald013/opt/miniconda3/bin/python

import cpuinfo

info = "\nCount:\t{0}\nBrand:\t{1}\nSpeed:\t{2}".format(cpuinfo.get_cpu_info()['count'], cpuinfo.get_cpu_info()['brand'], cpuinfo.get_cpu_info()['hz_actual'])
print(info)
