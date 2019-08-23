import json
with open('./experiments/veri776_res50_new/VeRi_70000_evaluation.json', 'r') as f:
    loaded_json = json.load(f)
cmc = loaded_json['CMC']
cmc = [x*100 for x in cmc]
res50_mslm = cmc

with open('./experiments/VeRi776_resnet101/VeRi_70000_evaluation.json', 'r') as f:
    loaded_json = json.load(f)
cmc = loaded_json['CMC']
cmc = [x*100 for x in cmc]
res101_mslm = cmc

with open('./experiments/veri776_mslm_mob3/VeRi_50000_evaluation.json', 'r') as f:
    loaded_json = json.load(f)
cmc = loaded_json['CMC']
cmc = [x*100 for x in cmc]
mob3_mslm = cmc


with open('./experiments/VeRi776_mobilenet_bs/VeRi_50000_evaluation.json', 'r') as f:
    loaded_json = json.load(f)

cmc = loaded_json['CMC']
cmc = [x*100 for x in cmc]
mob5_mslm = cmc


with open('VeRi_10000_evaluation_resnet.json', 'r') as f:
    loaded_json = json.load(f)
cmc = loaded_json['CMC']
cmc = [x*100 for x in cmc]
res50 = cmc

import matplotlib.pyplot as plt
import numpy as np

x=np.arange(0,len(cmc))
x= x+1
rank=30
plt.figure(1)
plt.plot(x[0:rank],res50[0:rank],color='r',linestyle='-',marker='^',linewidth=1,label='Res50+BH')
plt.plot(x[0:rank],mob3_mslm[0:rank],color='b',linestyle='-',marker='o',linewidth=1,label='MobV1+MSLM_3')
plt.plot(x[0:rank],mob5_mslm[0:rank],color='g',linestyle='-',marker='>',linewidth=1,label='MobV1+MSLM_5')
plt.plot(x[0:rank],res50_mslm[0:rank],color='y',linestyle='-',marker='d',linewidth=1,label='Res50+MSLM')
plt.plot(x[0:rank],res101_mslm[0:rank],color='m',linestyle='-',marker='*',linewidth=1,label='Res101+MSLM')

plt.xlabel('Rank')
plt.ylabel('Matching rate(%)')
plt.title('CMC evaluation for proposed models')
plt.legend()
plt.grid()
plt.savefig('cmc_for_veri776.eps',format='eps')
plt.show()

import json
with open('./experiments/veri776_res50_new/VeRi_70000_evaluation.json', 'r') as f:
    loaded_json = json.load(f)
cmc = loaded_json['CMC']
cmc = [x*100 for x in cmc]
res50_mslm = cmc

with open('./experiments/VeRi776_resnet101/VeRi_70000_evaluation.json', 'r') as f:
    loaded_json = json.load(f)
cmc = loaded_json['CMC']
cmc = [x*100 for x in cmc]
res101_mslm = cmc

with open('./experiments/veri776_mslm_mob3/VeRi_50000_evaluation.json', 'r') as f:
    loaded_json = json.load(f)
cmc = loaded_json['CMC']
cmc = [x*100 for x in cmc]
mob3_mslm = cmc


with open('./experiments/VeRi776_mobilenet_bs/VeRi_50000_evaluation.json', 'r') as f:
    loaded_json = json.load(f)

cmc = loaded_json['CMC']
cmc = [x*100 for x in cmc]
mob5_mslm = cmc


with open('VeRi_10000_evaluation_resnet.json', 'r') as f:
    loaded_json = json.load(f)
cmc = loaded_json['CMC']
cmc = [x*100 for x in cmc]
res50 = cmc

import matplotlib.pyplot as plt
import numpy as np

x=np.arange(0,len(cmc))
x= x+1
rank=30
plt.figure(1)
plt.plot(x[0:rank],res50[0:rank],color='r',linestyle='-',marker='^',linewidth=1,label='Res50+BH')
plt.plot(x[0:rank],mob3_mslm[0:rank],color='b',linestyle='-',marker='o',linewidth=1,label='MobV1+MSLM_3')
plt.plot(x[0:rank],mob5_mslm[0:rank],color='g',linestyle='-',marker='>',linewidth=1,label='MobV1+MSLM_5')
plt.plot(x[0:rank],res50_mslm[0:rank],color='y',linestyle='-',marker='d',linewidth=1,label='Res50+MSLM')
plt.plot(x[0:rank],res101_mslm[0:rank],color='m',linestyle='-',marker='*',linewidth=1,label='Res101+MSLM')

plt.xlabel('Rank')
plt.ylabel('Matching rate(%)')
plt.title('CMC evaluation for proposed models')
plt.legend()
plt.grid()
plt.savefig('cmc_for_veri776.eps',format='eps')
plt.show()
