ADDR1=172.17.20.237:2222
ADDR2=172.17.20.229:2223
INDEX?=0

run:
	TF_CONFIG='{"cluster": {"worker": ["${ADDR1}", "${ADDR2}"]}, "task": {"type": "worker", "index": ${INDEX}} }' python3.7 train.py