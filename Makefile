ADDR1=172.17.20.237:2222
ADDR2=172.17.20.229:2223
ADDR3=172.17.20.212:2224
ADDR4=172.17.20.144:2225
INDEX?=0

run:
	TF_CONFIG='{"cluster": {"worker": ["${ADDR1}", "${ADDR2}", "${ADDR3}", "${ADDR4}"]}, "task": {"type": "worker", "index": ${INDEX}} }' python3.7 train.py
