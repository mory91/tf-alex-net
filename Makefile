ADDR1=172.17.20.222:2222
ADDR2=172.17.20.116:2223
INDEX?=0

run:
	TF_CONFIG='{"cluster": {"worker": ["${ADDR1}", "${ADDR2}"]}, "task": {"type": "worker", "index": ${INDEX}} }' python train.py
