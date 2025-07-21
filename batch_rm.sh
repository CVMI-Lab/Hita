#!/bin/bash
alias oss="aws --endpoint-url=http://oss.i.basemind.com s3"
CUR_PATH=$(basename "${PWD}")
OSS_PATH=s3://whc/anlin/Paintmind/projects/hita-iccv
echo ${OSS_PATH}/${CUR_PATH}/
model_type='GPT-B'
for epoch in {1..123} 
do
    iters=`expr 10000 \* ${epoch}`
    # oss_file=${OSS_PATH}/${CUR_PATH}/output/ultra/${model_type}/${model_type}-${iters}.pt
    oss_file=s3://whc/anlin/Paintmind/projects/hita-iccv/hita-gpt/output/ultra/GPT-B/${model_type}-${iters}.pt
    a=`expr ${epoch} \% 5`
    b=`expr ${epoch} \% 10`
    if [ $a -eq 0 ] || [ $b -eq 0 ]; then
        continue
    fi
    # echo ${oss_file}
    aws --endpoint-url=http://oss.i.basemind.com s3 rm ${oss_file}
done
