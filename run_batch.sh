# !/bin/bash
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
OSS_PATH=s3://exp/Paintmind/projects/LlamaGen/AR-tokenizer.8.12dim.generator-base.cnn.depth.3.vqgan\@336pix/output/model_dump.firetooth
for i in {140..149}
do  
    iters=`expr $i \* 10000`
    cfg_scale='1.75'
    model_type='GPT-B'
    FILE=samples/${model_type}-${model_type}-${iters}-size-336-size-256-ViT-VQGAN-topk-0-topp-1.0-temperature-1.0-cfg-${cfg_scale}-seed-0.txt
    if [ -f ${FILE}.txt ]; then
        echo ${FILE}.txt
        continue
    fi
    # aws --endpoint-url=http://oss.i.brainpp.cn s3 cp s3://exp/Paintmind/projects/LlamaGen/AR-tokenizer.12dim.generator-large.cnn.8.12.d.vqgan@336pix/output/model_dump/GPT-L-${iters}.pt output/model_dump/
    model_file=output/model_dump/${model_type}-${iters}.pt
    echo ${model_file}
    if [ ! -f ${model_file} ]; then
        oss_file=${OSS_PATH}/${model_type}-${iters}.pt
        aws --endpoint-url=http://oss.i.brainpp.cn s3 cp ${oss_file} output/model_dump/
    fi
    bash run_test.sh ${iters} ${cfg_scale} 50
    if [ -f ${model_file} ]; then
        rm output/model_dump/${model_type}-${iters}.pt
    fi
done
