# !/bin/bash
alias oss="aws --endpoint-url=http://oss.i.basemind.com s3"
CUR_PATH=$(basename "${PWD}")
OSS_PATH=s3://whc/anlin/Paintmind/projects/hita-iccv
echo ${OSS_PATH}/${CUR_PATH}/
model_type='GPT-L'
if [ -d output/vanilla/${model_type} ]; then
      oss sync output/vanilla/${model_type} ${OSS_PATH}/${CUR_PATH}/vanilla/2machines/${model_type}
      rm output/vanilla/${model_type}/${model_type}-*.pt
fi
# oss cp ${CUR_PATH}.zip ${OSS_PATH}/${CUR_PATH}/
# if [ -f 'results.md' ];then
#     oss cp results.md ${OSS_PATH}/${CUR_PATH}/
# fi
# rm ${CUR_PATH}.zip
echo ${OSS_PATH}/${CUR_PATH}/
