
import os, pdb

def set_aws_a():
    os.environ['OSS_ENDPOINT'] = 'http://oss.i.basemind.com'
    os.environ['AWS_ACCESS_KEY_ID'] = 'cbd9cc9eaa437b626dc3973f5220f7a3'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'd409e0bfd0027e021b6b7707eeb3de3f'

def set_aws_b():
    os.environ['OSS_ENDPOINT'] = 'http://oss.i.shaipower.com'
    os.environ['AWS_ACCESS_KEY_ID'] = AK_B
    os.environ['AWS_SECRET_ACCESS_KEY'] = SK_B

def set_aws_all():
    os.environ['PROFILEA__OSS_ENDPOINT'] = 'http://oss.i.basemind.com'
    os.environ['PROFILEA__AWS_ACCESS_KEY_ID'] = AK_A
    os.environ['PROFILEA__AWS_SECRET_ACCESS_KEY'] = SK_A

    os.environ['PROFILEB__OSS_ENDPOINT'] = 'http://oss.i.shaipower.com'
    os.environ['PROFILEB__AWS_ACCESS_KEY_ID'] = AK_B
    os.environ['PROFILEB__AWS_SECRET_ACCESS_KEY'] = SK_B

def add_profile(oss_url, profile_type='a'):
    if profile_type == 'a':
        return 's3+profilea' + oss_url[len('s3'):]
    elif profile_type == 'b':
        return 's3+profileb' + oss_url[len('s3'):]
    else:
        raise NotImplementedError

def remove_profile(oss_url, profile_type='a'):
    if profile_type == 'a':
        return 's3' + oss_url[len('s3+profilea'):]
    elif profile_type == 'b':
        return 's3' + oss_url[len('s3+profileb'):]
    else:
        raise NotImplementedError

# 同时使用 ab机房aws
# 文档 https://megvii-research.github.io/megfile/configuration.html#config-for-different-s3-server-or-authentications