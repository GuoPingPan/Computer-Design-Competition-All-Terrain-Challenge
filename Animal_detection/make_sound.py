'''
	Create Sound by Baidu AipSpeech

	author: guopingpan
    email: 731061720@qq.com
           or panguoping02@gmail.com

	tutorials: https://ai.baidu.com/ai-doc/SPEECH/Gk38y8lzk

'''

from aip import AipSpeech

app_id = '25083603'
app_key = 'FYeGRhnSybAC0zerPy4GCPGa'
secret_key = '3mRcUNRP7Q8ETXUx0g3r2AdiKEXCuTRK'

client = AipSpeech(app_id,app_key,secret_key)

result = client.synthesis('羊','zh',1,{'vol':5,'per':1,'spd':4})

if not isinstance(result,dict):
	with open('play.mp3','wb') as f:
		f.write(result)
