rsync -avz \
    -e "ssh -i ~/.ssh/aws-antelligence-keys.pem" \
    --exclude 'dataset/celeba-resnet-public.hdf5' \
    --exclude 'dataset/agnews-mxbai-public.hdf5' \
    --exclude 'dataset/gooaq-distilroberta-public.hdf5' \
    --exclude 'dataset/imagenet-clip-public.hdf5' \
    --exclude 'dataset/landmark-nomic-public.hdf5' \
    --exclude 'dataset/simplewiki-openai-public.hdf5' \
    --exclude 'dataset/yahoo-minilm-public.hdf5' \
    --exclude '__pycache__/*' \
    --exclude '*.egg-info' \
    --exclude '.venv' \
    --exclude '.git' \
    . ec2-user@ec2-13-60-193-163.eu-north-1.compute.amazonaws.com:/home/ec2-user/ANNvedi

